from collections import OrderedDict
from datetime import datetime
from glob import glob
from os import listdir, makedirs
from os.path import basename, isfile, join
from time import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from efficientvit.sam_model_zoo import create_sam_model
from module.preprocess import preprocess, resize_longest_side, pad_image
from module.argparser import creater_parser
from models.medsam_lite import MedSAM_Lite
from module.visualize import draw_result_2d, draw_result_3d, save_pred
from module.inference import MedSAM_infer_npz_2D, MedSAM_infer_npz_3D




# %% set seeds
torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

args = creater_parser()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
if save_overlay:
    assert (
        args.png_save_dir is not None
    ), "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256



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


medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64,  ## (64, 256, 256)
        128,  ## (128, 128, 128)
        160,  ## (160, 64, 64)
        320,  ## (320, 64, 64)
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8,
)

efficientvit_sam = create_sam_model(
name="l0", weight_url="./l0.pt",
)
medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16,
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    ),
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder=efficientvit_sam.image_encoder,
    # image_encoder=medsam_lite_image_encoder,
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder,
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location="cpu")
medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
# medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()


if __name__ == "__main__":
    img_npz_files = sorted(glob(join(data_root, "*.npz"), recursive=True))
    efficiency = OrderedDict()
    efficiency["case"] = []
    efficiency["time"] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        npz_name = basename(img_npz_file)
        if basename(img_npz_file).startswith("3D"):
            segs, boxes_3D, img_3D = MedSAM_infer_npz_3D(medsam_lite_model,img_npz_file, device)
            if save_overlay:
                draw_result_3d(img_3D, boxes_3D, segs, png_save_dir)

        else:
            segs, boxes, img_3c = MedSAM_infer_npz_2D(medsam_lite_model, img_npz_file, device)
            if save_overlay:
                draw_result_2d(img_3D, boxes_3D, segs, png_save_dir)
        save_pred(pred_save_dir, npz_name, segs)
        end_time = time()
        efficiency["case"].append(basename(img_npz_file))
        efficiency["time"].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            current_time,
            "file name:",
            basename(img_npz_file),
            "time cost:",
            np.round(end_time - start_time, 4),
        )
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, "efficiency.csv"), index=False)
