import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor="blue"):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2)
    )

def save_pred(pred_save_dir, npz_name, segs):
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
def draw_result_2d(img_3c, boxes, segs, png_save_dir, npz_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3c)
    ax[1].imshow(img_3c)
    ax[0].set_title("Image")
    ax[1].set_title("LiteMedSAM Segmentation")
    ax[0].axis("off")
    ax[1].axis("off")

    for i, box in enumerate(boxes):
        color = np.random.rand(3)
        box_viz = box
        show_box(box_viz, ax[1], edgecolor=color)
        show_mask((segs == i + 1).astype(np.uint16), ax[1], mask_color=color)

    plt.tight_layout()
    plt.savefig(join(png_save_dir, npz_name.split(".")[0] + ".png"), dpi=300)
    plt.close()



def draw_result_3d(img_3D, boxes_3D, segs, png_save_dir, npz_name):
    idx = int(segs.shape[0] / 2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3D[idx], cmap="gray")
    ax[1].imshow(img_3D[idx], cmap="gray")
    ax[0].set_title("Image")
    ax[1].set_title("LiteMedSAM Segmentation")
    ax[0].axis("off")
    ax[1].axis("off")

    for i, box3D in enumerate(boxes_3D, start=1):
        if np.sum(segs[idx] == i) > 0:
            color = np.random.rand(3)
            x_min, y_min, z_min, x_max, y_max, z_max = box3D
            box_viz = np.array([x_min, y_min, x_max, y_max])
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask(segs[idx] == i, ax[1], mask_color=color)

    plt.tight_layout()
    plt.savefig(join(png_save_dir, npz_name.split(".")[0] + ".png"), dpi=300)
    plt.close()