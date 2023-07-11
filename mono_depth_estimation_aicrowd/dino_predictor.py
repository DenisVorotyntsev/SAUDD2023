import time
import torch
import numpy as np

from mono_depth_estimation_aicrowd.utils import Clock
from mono_depth_estimation_aicrowd.my_models import DinoModel
from mono_depth_estimation_aicrowd.predict_utils import make_prediction_dino
from mono_depth_estimation_aicrowd.preprocess import get_preprocess_for_dino


class DinoPredictor:
    def __init__(self):
        """
        Initialize the DinoPredictor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_paths = [
            "./mono_depth_estimation_aicrowd/models/fold_0/model.ckpt",
            "./mono_depth_estimation_aicrowd/models/fold_1/model.ckpt",
            "./mono_depth_estimation_aicrowd/models/fold_2/model.ckpt",
            # I din't have compute power to train full 5 folds
            # './mono_depth_estimation_aicrowd/models/fold_3/model.ckpt',
            # './mono_depth_estimation_aicrowd/models/fold_4/model.ckpt',
        ]
        self.models = []
        for model_path in self.models_paths:
            model = DinoModel()
            model_weight = torch.load(model_path)["model"]
            model.load_state_dict(model_weight)
            model.eval()
            self.models.append(model)

        self.transform_func = get_preprocess_for_dino()
        self.clock = Clock(10)

    def raise_aicrowd_error(self, msg: str):
        """Will be used by the evaluator to provide logs, DO NOT CHANGE"""
        raise NameError(msg)

    @torch.no_grad()
    def predict_depth_single_image(
        self, image_to_predict_depth: np.ndarray
    ) -> np.ndarray:
        """
        Predict the depth of a single image.

        Args:
            image_to_predict_depth (np.ndarray): The image for depth prediction.

        Returns:
            np.ndarray: The depth map.
        """
        self.clock.tik()

        # Preprocess data first
        midas_input_rgb = image_to_predict_depth / 255.0
        if midas_input_rgb.ndim == 2:
            midas_input_rgb = np.stack((midas_input_rgb,) * 3, axis=-1)
        t_image = self.transform_func(
            {
                "image": midas_input_rgb,
                "mask": midas_input_rgb,
                "depth": midas_input_rgb,
            }
        )["image"]
        t_image = torch.Tensor(t_image).to(self.device).unsqueeze(0)  # bs=1, 3, H, W

        # Run several models for prediction
        time_per_prediction = None
        n_models_so_far = 0

        ans = np.zeros(image_to_predict_depth.shape)
        for _, model in enumerate(self.models):
            t_start_pred = time.time()

            model.to(self.device)

            # predict
            disparity_image = model(t_image)

            disparity_image = torch.nn.functional.interpolate(
                disparity_image,
                size=tuple(image_to_predict_depth.shape),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            disparity_image = disparity_image.squeeze(0)
            disparity_image = disparity_image.cpu().numpy()

            ans += disparity_image
            n_models_so_far += 1
            t_end_pred = time.time()

            # free gpu mem
            model.cpu()

            delta_time = t_end_pred - t_start_pred
            if time_per_prediction is None:
                time_per_prediction = delta_time
            else:
                time_per_prediction = max([time_per_prediction, delta_time])

            time_left = self.clock.tok()
            if time_per_prediction * 1.25 > time_left:
                break

        ans = ans / n_models_so_far
        ans = np.clip(ans, 1e-8, 1_000)
        return ans
