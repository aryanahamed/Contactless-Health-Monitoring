# segmentation.py
import numpy as np
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import ImageSegmenter
from mediapipe.tasks.python.vision import image_segmenter
import mediapipe as mp
import core.config as config

# how do I use gpu delegate for this


class Segmenter:
    def __init__(self):
        try:
            # Attempt GPU delegate first
            base_options = mp_python.BaseOptions(
                model_asset_path=config.get_segmentation_model_path(),
                delegate=mp_python.BaseOptions.Delegate.GPU
            )
            # Try creating segmenter with GPU options to force check
            temp_options = image_segmenter.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
            ImageSegmenter.create_from_options(temp_options).close()  # Test creation & close immediately
        except Exception as e:
            base_options = mp_python.BaseOptions(
                model_asset_path=config.get_segmentation_model_path(),
                delegate=mp_python.BaseOptions.Delegate.CPU
            )

        # Now create the final options and the segmenter instance
        options = image_segmenter.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,  # Ensure this is True
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.segmenter = ImageSegmenter.create_from_options(options)

    def get_roi_mask(self, mp_img,timestamp,w,h):
        if mp_img is None:
            return None
        # Convert BGR to RGB
        target_labels = [1, 2, 3]

        # Segment the frame
        result= self.segmenter.segment_for_video(mp_img, int(timestamp * 1000))

        if result is None or result.category_mask is None:
            return None  # No segmentation

        label_mask = result.category_mask.numpy_view().astype(np.uint8)  # 256x256 mask
        roi_mask_resized = cv2.resize(label_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return roi_mask_resized



