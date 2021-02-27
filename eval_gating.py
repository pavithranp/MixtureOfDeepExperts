import random

# import some common detectron2 utilities
# from detectron2.utils.visualizer import Visualizer
from dataloader.single_loader import SingleDataset
# from tqdm import tqdm
import cv2 ,os
import math,torch
import numpy as np
import operator
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataloader.dataset import Dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from utils.eval import evaluate,readAndSortBBs,get_dicts, read_gt
import pickle
from network.GatingNetwork import  GatingNetwork

if __name__ == "__main__":

    val_dataset = SingleDataset(root='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD',dataset='DepthJetQhd',
                                    set='test',grad=True)
    sortedListTestBB =[]
    # args="ImagesQ_hd/" # change to ImagesQhd/
    args = "DepthJetQhd/"
    # x = Dataset(args)
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = 'output/'
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'output/model_0019999.pth'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.WEIGHTS = 'output/model_0019999.pth'
    cfg2 = cfg.clone()
    cfg2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg2.MODEL.WEIGHTS = 'output/rgb.pth'


    outputs = []
    predictions = {}
    # dataset_dicts = x.load_annotations()
    print("validating with:",len(val_dataset.files))
    with torch.no_grad():
        gn = GatingNetwork(cfg, cfg2)
        gn.eval()
        for d in val_dataset.files:
            rgb = cv2.imread(os.path.join(val_dataset.root, 'ImagesQ_hd', d + '.png'))
            depth = cv2.imread(os.path.join(val_dataset.root,'DepthJetQhd',d+'.png'))
            image1 = torch.as_tensor(rgb.transpose(2, 0, 1), dtype=torch.float32)
            image2 = torch.as_tensor(depth.transpose(2, 0, 1), dtype=torch.float32)
            output = gn([{'rgb_image':image1,'depth_image':image2,'height': 540,'width':960 }])
            pred = get_dicts(d,output[0])
            predictions[pred[0]]= [pred[1],pred[2]]

    # print(outputs)
    groundtruth_boxes = read_gt()
    number_of_groundtruth_boxes = sum([len(groundtruth_boxes[x]) for x in groundtruth_boxes.keys() if x in predictions.keys()])
    j,k = readAndSortBBs(predictions,groundtruth_boxes,sortedListTestBB)
    evaluate(j, groundtruth_boxes, number_of_groundtruth_boxes, threshold=0.6)

    SOFTMAX_THRESHOLD = 0.6







