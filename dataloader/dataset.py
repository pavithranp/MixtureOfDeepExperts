# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import os
import numpy as np
import yaml


class Dataset():
    def __init__(self, args):
        self.dataset_name = 'InOutDoor'
        self._dataset = args.dataset_path
        self._image_set_path = args.image_set_path
        self._image_set = args.image_sets
        self._dataset = args.dataset_path
        self._annotation_path = args.annotations
        self.dataset_dicts = []

    def load_annotations(self,set):
        with open(self._image_set_path + self._image_set + '.yml') as annon_file:
            for line in annon_file:
                x = self.read_annon_file(self._annotation_path + line + '.yml')
                record = {}
                record["file_name"] = x['filename']
                record["image_id"] = x['filename'][-4]
                record["height"] = 1920
                record["width"] = 1080

                objects = x['objects']
                objs = []
                for _, anno in objects:
                    bndbox = anno["bndbox"]

                    obj = {
                        "bbox": [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "objectness_logits": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
            self.dataset_dicts.append(record)

    def register_dataset(self):
        for d in ["train", "val"]:
            self._image_set = d
            DatasetCatalog.register(self.dataset_name+"_" + d, lambda d=d: self.load_annotations(d))
            MetadataCatalog.get(self.dataset_name+"_" + d).set(thing_classes=["Pedestrians"])
        balloon_metadata = MetadataCatalog.get("balloon_train")

    @staticmethod
    def read_annon_file(file):
        skip_lines = 2
        with open(file) as infile:
            for i in range(skip_lines):
                _ = infile.readline()
            return yaml.load(infile, Loader=yaml.FullLoader)

    # dataset visualizer
