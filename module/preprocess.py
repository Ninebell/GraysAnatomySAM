import numpy as np
import cv2
def anisotropic_diffusion(img, num_iter, kappa, gamma=0.1):
    """
    Apply anisotropic diffusion (Perona-Malik filter) to an image.

    Parameters:
        img (numpy.ndarray): The input image (grayscale).
        num_iter (int): Number of iterations to run the diffusion.
        kappa (float): Conductance coefficient, controls diffusion amount.
        gamma (float): Integration constant (small timestep).

    Returns:
        numpy.ndarray: The image after applying anisotropic diffusion.
    """
    # Convert image to float and normalize to range 0-1
    img = img.astype(np.float32) / 255

    # Initialize the output image
    diffused_img = img.copy()

    for i in range(num_iter):
        # Calculate gradients
        north = np.roll(diffused_img, -1, axis=0)
        south = np.roll(diffused_img, 1, axis=0)
        east = np.roll(diffused_img, 1, axis=1)
        west = np.roll(diffused_img, -1, axis=1)

        # Compute differences
        delta_n = north - diffused_img
        delta_s = south - diffused_img
        delta_e = east - diffused_img
        delta_w = west - diffused_img

        # Calculate the diffusion flux
        c_n = np.exp(-((delta_n / kappa) ** 2))
        c_s = np.exp(-((delta_s / kappa) ** 2))
        c_e = np.exp(-((delta_e / kappa) ** 2))
        c_w = np.exp(-((delta_w / kappa) ** 2))

        # Update the image
        diffused_img += gamma * (
            c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w
        )
    # Rescale back to 8-bit image values
    return (diffused_img * 255).astype(np.uint8)
def rgb2gray(rgb):
    return np.array(
        np.round(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])), dtype="uint8"
    )


def preprocess(img):
    img = (img-img.min())/(img.max()-img.min()) * 255
    gray_img = rgb2gray(img)

    diffused_img = anisotropic_diffusion(gray_img, num_iter=1, kappa=20)
    equalized_img = cv2.equalizeHist(gray_img)
    merged_img = cv2.merge((gray_img, diffused_img, equalized_img))
    return merged_img



def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:  ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded