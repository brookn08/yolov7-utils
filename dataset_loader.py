import logging
import random
from pathlib import Path
from typing import List, Optional

from cvat_utils.tlnk_cvat_manipulator import (CvatFrameObject,
                                              TlnkCvatManipulator)

logger = logging.getLogger("Dataset loading script")

class DatasetLoader():
    def __init__(self, path: Path, shuffle: bool = True):
        self.path = Path(path)
        self.data: List[CvatFrameObject] = []
        self._update_data_list()
        self.shuffle_data(shuffle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        frame_object = self.data[index]
        img_path = frame_object.get_image_name()
        img = frame_object.get_img()
        label = frame_object.get_body_bboxes()

        return img_path, label, img.shape

    def shuffle_data(self, shuffle: Optional[bool] = True):
        if shuffle:
            random.shuffle(self.data)
        return

    def get_data(self):
        return self.data

    def _update_data_list(self):
        for project_path in self.path.iterdir():
            logger.info("extracting image path and corresponding annotations from {}".format(project_path))
            if project_path.is_dir():
                self._update_project_data(project_path)
        return

    def _update_project_data(self, project_path: Path):
        for task_path in project_path.iterdir():
            if task_path.is_dir():
                self._update_task_data(task_path)
        return

    def _update_task_data(self, task_path: Path):
        cvat_manipulator = TlnkCvatManipulator(task_path)
        self.data += cvat_manipulator.get_all_images_bodies()
        return