import numpy as np
import cv2

def applyAdjustedColorToSkinRegion(target_skin, target_skin_mask, adjusted_color):
    """
    Apply the adjusted color to the skin region of the target image.

    Args:
        target_skin (numpy.ndarray): The target image.
        target_skin_mask (numpy.ndarray): The mask indicating the skin region.
        adjusted_color (tuple or numpy.ndarray): The adjusted color to apply.

    Returns:
        numpy.ndarray: The target image with the adjusted color applied to the skin region.
    """
    # Make a copy of the target_skin to avoid modifying the original image
    target_skin_result = target_skin.copy()

    # Iterate over each pixel in the target_skin
    for row in range(target_skin.shape[0]):
        for col in range(target_skin.shape[1]):
            # Check if the pixel is part of the skin region
            if target_skin_mask[row, col]:
                # Apply the adjusted color to the skin region
                target_skin_result[row, col] = adjusted_color

    return target_skin_result



def blendSkinWithTexture(target_skin,target_skin_mask,target_skin_result,resized_target_image, alpha=0.5):
    """
    Blend skin texture with the original target skin.

    Args:
        target_skin (numpy.ndarray): Original target skin image.
        target_skin_mask (numpy.ndarray): Mask indicating the skin region.
        target_skin_result (numpy.ndarray): Image with adjusted color for skin.
        resized_target_image (numpy.ndarray): Resized target image.
        alpha (float): Blending parameter (default is 0.5).

    Returns:
        numpy.ndarray: Blended image.
    """
    # Apply bilateral filter for texture separation
    kernel_size = 5
    sigma_space = sigma_color = 10
    target_texture = cv2.bilateralFilter(target_skin, kernel_size, sigma_space, sigma_color)

    # Subtract low-frequency component to isolate high-frequency texture
    target_texture = cv2.subtract(target_skin, target_texture)

    # Optionally, normalize texture values (0-255)
    target_texture = cv2.normalize(target_texture, None, 0, 255, cv2.NORM_MINMAX)

    # Create alpha channel from mask
    alpha_channel = target_skin_mask * 255  # Convert mask to 8-bit alpha channel

    # Blend using alpha blending
    blended_texture = cv2.addWeighted(target_texture, alpha,target_skin_result, alpha, 0, dtype=cv2.CV_8U)

    # Blend the result with the resized target image
    blended_image = cv2.addWeighted(resized_target_image, alpha, blended_texture, alpha, 0, dtype=cv2.CV_8U)

    return blended_image

