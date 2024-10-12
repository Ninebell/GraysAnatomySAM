
from os.path import basename, isfile, join
import numpy as np
from module.preprocess import preprocess, resize_longest_side, pad_image
import torch

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(
        box_256[None, None, ...], dtype=torch.float, device=img_embed.device
    )

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = medsam_model.postprocess_masks(
        low_res_logits, new_size, original_size
    )
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint16)

    return medsam_seg, iou

def MedSAM_infer_npz_2D(medsam_lite_model, img_npz_file, device):
    npz_data = np.load(img_npz_file, "r", allow_pickle=True)  # (H, W, 3)
    img_3c = npz_data["imgs"]  # (H, W, 3)
    # preprocess image
    # gray_img = rgb2gray(img_3c)
    # diffused_img = anisotropic_diffusion(gray_img, num_iter=1, kappa=20)
    # equalized_img = cv2.equalizeHist(gray_img)
    # img_3c = cv2.merge((gray_img, diffused_img, equalized_img))
    img_3c = preprocess(img_3c)
    assert (
        np.max(img_3c) < 256
    ), f"input data should be in range [0, 255], but got {np.unique(img_3c)}"
    H, W = img_3c.shape[:2]
    boxes = npz_data["boxes"]
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint16)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = (
        torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...]  # (1, 4)
        medsam_mask, iou_pred = medsam_inference(
            medsam_lite_model, image_embedding, box256, (newh, neww), (H, W)
        )
        segs[medsam_mask > 0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
    return segs, boxes, img_3c


def MedSAM_infer_npz_3D(medsam_lite_model, img_npz_file, device):
    npz_data = np.load(img_npz_file, "r", allow_pickle=True)
    img_3D = npz_data["imgs"]  # (D, H, W)
    spacing = npz_data[
        "spacing"
    ]  # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint16)
    boxes_3D = npz_data["boxes"]  # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert (
            z_min < z_max
        ), f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min) / 2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            # preprocess image
            img_3c = preprocess(img_3c)
            # gray_img = rgb2gray(img_3c)
            # diffused_img = anisotropic_diffusion(gray_img, num_iter=1, kappa=20)
            # equalized_img = cv2.equalizeHist(gray_img)
            # img_3c = cv2.merge((gray_img, diffused_img, equalized_img))
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            # convert the shape to (3, H, W)
            img_256_tensor = (
                torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(
                    img_256_tensor
                )  # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs[z - 1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(
                medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W]
            )
            segs_3d_temp[z, img_2d_seg > 0] = idx

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle - 1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            # preprocess image
            img_3c = preprocess(img_3c)
            # gray_img = rgb2gray(img_3c)
            # diffused_img = anisotropic_diffusion(gray_img, num_iter=1, kappa=20)
            # equalized_img = cv2.equalizeHist(gray_img)
            # img_3c = cv2.merge((gray_img, diffused_img, equalized_img))
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = (
                torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(
                    img_256_tensor
                )  # (1, 256, 64, 64)

            pre_seg = segs[z + 1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                scale_256 = 256 / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            img_2d_seg, iou_pred = medsam_inference(
                medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W]
            )
            segs_3d_temp[z, img_2d_seg > 0] = idx
        segs[segs_3d_temp > 0] = idx
    return segs, boxes_3D, img_3D
 