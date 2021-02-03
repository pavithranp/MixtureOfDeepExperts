from __future__ import print_function, division
import yaml
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from torchvision import transforms, utils

import os ,cv2
import numpy as np
import torch
from detectron2.data import detection_utils as utils


class RGBD(object):
    def __init__(self, root, transforms =None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.rgb_imgs = list(sorted(os.listdir(os.path.join(root, "ImagesQ_hd"))))
        self.annon_imgs = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "DepthJetQhd"))))
        self.files=[]
        self._image_set_path = os.path.join(root, "ImageSets")
        with open(os.path.join(self._image_set_path, 'train.txt')) as annon_file:
            for x in annon_file:
                annon = self.read_annon_file(os.path.join(root,'Annotations',x[:-1]+'.yml'))
                if 'object' in annon:
                    self.files.append(x[:-1])

    def __getitem__(self, idx):
        # load images ad masks
        rgb_path = os.path.join(self.root, "ImagesQhd", self.files[idx]+'.png')
        depth_path = os.path.join(self.root, "DepthJetQhd", self.files[idx]+'.png')
        annon_path = os.path.join(self.root, "Annotations", self.files[idx]+'.yml')
        rgb = self.image_process(rgb_path)
        depth = self.image_process(depth_path)
        annon = self.read_annon_file(annon_path)
        # convert the PIL Image into a numpy array

        # get bounding box coordinates for each mask
        # if 'object' in annon:
        objects = annon['object']

        num_objs = len(objects)
        boxes  = []
        ratiox = 960/1920
        ratioy = 540/1080
        for anno in objects:
            bndbox = anno["bndbox"]
            boxes.append([float(bndbox['xmin']) * ratiox, float(bndbox['ymin']) * ratioy,
                         float(bndbox['xmax']) * ratiox, float(bndbox['ymax']) * ratioy])
            # objs.append(obj)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        # targets = utils.annotations_to_instances(objs, (540, 960))
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            rgb, target = self.transforms(rgb, target)

        return {'rgb_image':rgb, 'depth_image':depth , 'height':540, 'width': 960, 'target':  target}

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(batch):
        for x in batch:
            objs = []
            for box in x['target']['boxes']:
                obj = {
                    "bbox": [box[0], box[1], box[2], box[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 1
                }
                objs.append(obj)
            x['instances'] = utils.annotations_to_instances(objs, (540, 960))
        return batch  # Show how to implement a minimal mapper, similar to the default DatasetMapper

    @staticmethod
    def read_annon_file(file):
        skip_lines = 2
        with open(file) as infile:
            for i in range(skip_lines):
                _ = infile.readline()
            return yaml.load(infile, Loader=yaml.FullLoader)

    @staticmethod
    def image_process(path):
        image = cv2.imread(path)
        image = torch.tensor(image.transpose(2, 0, 1), requires_grad=True, dtype=torch.float32)
        return image