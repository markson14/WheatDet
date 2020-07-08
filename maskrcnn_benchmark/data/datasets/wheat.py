import os

import torch
import torch.utils.data
from PIL import Image
import sys
import pandas as pd

from maskrcnn_benchmark.structures.bounding_box import BoxList


class WheatDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "wheat",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "train_gai.csv")
        self._imgpath = os.path.join(self.root, "train", "%s")
        # self._imname = os.listdir(os.path.join(self.root, "train")) # for selecting no_label
        if split == "train":
            self._imname = os.listdir(os.path.join(self.root, "train"))[:-665]
        else:
            self._imname = os.listdir(os.path.join(self.root, "train"))[-665:]
        self._anno = pd.read_csv(self._annopath)

        cls = WheatDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self._imname[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target.add_field("ID", self._imgpath % self._imname[index])
        return img, target, index

    def __len__(self):
        return len(self._imname)

    def get_groundtruth(self, index):
        img_id = self._imname[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(img_id)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, img_id):
        boxes = []
        gt_classes = []
        TO_REMOVE = 1

        img_name = img_id.split(".")[0]
        records = self._anno[self._anno['image_id'] == img_name]

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        height = records[['height']].values[0]
        width = records[['width']].values[0]

        im_info = tuple(map(int, (height, width)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((records.shape[0],), dtype=torch.int64),
            "difficult": torch.zeros((records.shape[0],), dtype=torch.int64),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self._imname[index]
        img_name = img_id.split(".")[0]
        records = self._anno[self._anno['image_id'] == img_name]
        return {"height": records[['height']].values[0], "width": records[['width']].values[0], "ID": self._imgpath % self._imname[index]}

    def map_class_id_to_class_name(self, class_id):
        return WheatDataset.CLASSES[class_id]


if __name__ == "__main__":
    import shutil
    dataset = WheatDataset('/home/zhangziwei/wheat_det/', 'valid')
    print(dataset.get_img_info(0))