from detectron2.engine import DefaultTrainer
import os
# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from dataloader.dataset import Dataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
if __name__=="__main__":
    args="DepthJetQhd/" # change to ImagesQhd/
    x = Dataset(args,set='val')
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_val",)
    print("InOutDoorDepth_val")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = 'output'
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'output/rgb.pth'
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    # x.visualize()
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    model = build_model(cfg)
    checkpointer2 = DetectionCheckpointer(model)
    checkpointer2.load(cfg.MODEL.WEIGHTS)
    model.eval()
    evaluator = COCOEvaluator("InOutDoorDepth_val", ("bbox"), False, output_dir="output/")
    val_loader = build_detection_test_loader(cfg, "InOutDoorDepth_val")
    print(inference_on_dataset(model, val_loader, evaluator))