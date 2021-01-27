# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

import yaml
import cv2, random



class Dataset():
    def __init__(self, args):
        self.dataset_name = 'InOutDoorRGB'
        self._dataset = "/no_backups/d1386/InOutDoorPeopleRGBD/Images/"
        self._image_set_path = "/no_backups/d1386/InOutDoorPeopleRGBD/ImageSets/"
        self._image_set = 'train'
        self._annotation_path = "/no_backups/d1386/InOutDoorPeopleRGBD/Annotations/"

        #        self._dataset = args.dataset_path
        #        self._image_set_path = args.image_set_path
        #        self._image_set = args.image_sets
        #        self._dataset = args.dataset_path
        #        self._annotation_path = args.annotations
        self.dataset_dicts = []
        self.register_dataset()

    def load_annotations(self):
        # self._image_set = set
        with open(self._image_set_path + self._image_set + '.txt') as annon_file:
            size =0
            for line in annon_file:
                x = self.read_annon_file(self._annotation_path + line[:-1] + '.yml')
                record = {"image_id": x['filename'][:-4]}
                record["file_name"] = self._dataset + record['image_id']+'.png'
                record["height"] = 1080
                record["width"] = 1920
                if not 'object' in x:
                    continue
                size +=1
                objects = x['object']
                objs = []
                for anno in objects:
                    bndbox = anno["bndbox"]
                    obj = {
                        "bbox": [float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        # "objectness_logits": 1.0,
                        "category_id": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                self.dataset_dicts.append(record)
            print("Dataset contains : ", size, "elements for ",self._image_set)
            return self.dataset_dicts

    def register_dataset(self):
        for d in ["train"]:
            self._image_set = d
            DatasetCatalog.register(self.dataset_name + "_" + d, self.load_annotations)
            MetadataCatalog.get(self.dataset_name + "_" + d).set(thing_classes=["Pedestrians"])

    @staticmethod
    def read_annon_file(file):
        skip_lines = 2
        with open(file) as infile:
            for i in range(skip_lines):
                _ = infile.readline()
            return yaml.load(infile, Loader=yaml.FullLoader)

    # dataset visualizer

    def visualize(self):
        balloon_metadata = MetadataCatalog.get("self.dataset_name" + "_" +"train")
        dataset_dicts = self.load_annotations()
        for d in random.sample(dataset_dicts, 2):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('test',out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    args = []
    # x.load_annotations('train')



