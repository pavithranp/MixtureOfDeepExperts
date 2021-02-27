from detectron2.engine import DefaultTrainer
import os
# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from dataloader.dataset import Dataset
import argparse
def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--modality_path', default= "DepthJetQhd/",
                        help='path to RGB model')
    parser.add_argument('--batch_size',  default=2,
                        help='batch size for dataloader')
    parser.add_argument('--workers',  default=4,
                        help='no of workers for dataloader')
    parser.add_argument('--iterations', default=10000,
                        help='no. of iterations for training')
    parser.add_argument('--out_dir', default='RGBD',
                        help='output directory to save models')
    return parser.parse_args()

if __name__=="__main__":

    args = parse_arg()
     # change to ImagesQhd/
    x = Dataset(args.modality_path)
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.SOLVER.MAX_ITER = args.iterations
    cfg.OUTPUT_DIR = args.modality_path
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 20   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    os.makedirs(args.modality_path, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
