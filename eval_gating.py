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
import argparse
from network.GatingNetwork import  GatingNetwork

def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model1', default= "output/model_final.pth",
                        help='path to RGB model')
    parser.add_argument('--model2', default="output/model_0019999.pth",
                        help='path to Depth model')
    parser.add_argument('--gated', default="output/RGBD.pth",
                        help='path to gated model')
    parser.add_argument('--data',  default='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD',
                        help='path to InOutDoorData')
    parser.add_argument('--out_dir', default='RGBD',
                        help='output directory to save models')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arg()
    val_dataset = SingleDataset(root=args.data,dataset='DepthJetQhd',
                                    set='test',grad=True)
    sortedListTestBB =[]

    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = 'output/'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = args.model1
    cfg2 = cfg.clone()
    cfg2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg2.MODEL.WEIGHTS = args.model2


    outputs = []
    predictions = {}
    # dataset_dicts = x.load_annotations()
    print("validating with:",len(val_dataset.files))
    with torch.no_grad():
        gn = GatingNetwork(cfg, cfg2)
        gn.set_training(False)
        wb = torch.load(args.gated)
        gn.weight = wb["weights"]
        gn.bias = wb["bias"]

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







