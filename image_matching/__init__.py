"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""

import numpy as np
from PIL import Image
from image_matching import viz2d
import torch
import cv2
import matplotlib.pyplot as plt
import os

__version__ = "0.1.0"

from .utils import supress_stdout

available_models = [
    "superglue",
]


def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch


@supress_stdout
def get_matcher(
    matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs
):
    if isinstance(matcher_name, list):
        from image_matching.base_matcher import EnsembleMatcher

        return EnsembleMatcher(matcher_name, device, *args, **kwargs)
    if matcher_name == "superglue":
        from image_matching.matching_toolbox import SuperGlueMatcher

        return SuperGlueMatcher(device, max_num_keypoints, *args, **kwargs)

    else:
        raise RuntimeError(
            f"Matcher {matcher_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )


device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def match_images(
    rgb_image: str,
    ir_image: str,
    output_dir: str = None,
    img_size: int = 512,
) -> dict:
    # Load images
    rgb_img = Image.open(rgb_image)
    thermal_img = Image.open(ir_image)

    matcher = get_matcher("superglue", device=device)

    # Convert PIL images to numpy arrays and resize
    img0_np = np.array(rgb_img.resize((img_size, img_size)))
    img1_np = np.array(thermal_img.resize((img_size, img_size)))

    # Convert numpy arrays to torch tensors
    img0 = torch.from_numpy(img0_np).float().permute(2, 0, 1) / 255.0
    img1 = torch.from_numpy(img1_np).float().permute(2, 0, 1) / 255.0

    # Perform matching
    result = matcher(img0, img1)
    num_inliers, H, inlier_kpts0, inlier_kpts1 = (
        result["num_inliers"],
        result["H"],
        result["inlier_kpts0"],
        result["inlier_kpts1"],
    )

    if H is not None and num_inliers > 10:
        # Visualize the results
        plt.subplots(1, 2, figsize=(20, 10))
        viz2d.plot_images([img0, img1])
        viz2d.plot_matches(inlier_kpts0, inlier_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f"{num_inliers} matches", fs=20)
        # save to file
        if output_dir:
            plt.savefig(output_dir + "/matches.png", format="PNG")

        # Warp img0 to align with img1
        img0_warped = cv2.warpPerspective(img0_np, H, (img_size, img_size))

        # Create a mask for the warped image
        mask = (img0_warped != 0).all(axis=2).astype(np.uint8) * 255

        # Add an alpha channel to the warped image
        img0_warped_rgba = cv2.cvtColor(img0_warped, cv2.COLOR_RGB2RGBA)
        img0_warped_rgba[:, :, 3] = mask

        # Convert img1 to RGBA
        img1_rgba = cv2.cvtColor(img1_np, cv2.COLOR_RGB2RGBA)

        # Blend the images
        alpha_overlay = 0.5  # Adjust this value to change the transparency level
        blended = cv2.addWeighted(
            img1_rgba, alpha_overlay, img0_warped_rgba, 1 - alpha_overlay, 0
        )

        # Save the blended image
        if output_dir:
            Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_RGBA2RGB)).save(
                output_dir + "/matches_blended.png", format="PNG"
            )

        return {
            "num_inliers": num_inliers,
            "H": H,
            "img_size": img_size,
        }
    else:
        raise {"message": f"Not enough inliers found: {num_inliers}"}
