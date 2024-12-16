import sys
from pathlib import Path
import yaml
import numpy as np
import torchvision.transforms as tfm

BASE_PATH = Path(__file__).parent.parent.resolve() / "image_matching" / "superglue"
sys.path.append(str(Path(BASE_PATH)))
from image_matching.immatch import SuperGlue
from image_matching.base_matcher import BaseMatcher


class SuperGlueMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.to_gray = tfm.Grayscale()

        with open(BASE_PATH.joinpath("superglue.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["sat"]
        args["max_keypoints"] = max_num_keypoints

        self.matcher = SuperGlue(args)

        # move models to proper device - immatch reads cuda available and defaults to GPU
        self.matcher.model.to(device)  # SG
        self.matcher.detector.to(device)  # SP

        self.match_threshold = args["match_threshold"]
        # print(self.matcher.detector.model.config)

    def _forward(self, img0, img1):

        img0_gray = self.to_gray(img0).unsqueeze(0).to(self.device)
        img1_gray = self.to_gray(img1).unsqueeze(0).to(self.device)

        matches, kpts0, kpts1, _ = self.matcher.match_inputs_(img0_gray, img1_gray)
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        return mkpts0, mkpts1, kpts0, kpts1, None, None
