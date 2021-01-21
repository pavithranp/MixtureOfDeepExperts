# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
# import random
from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch ,cv2
import numpy as np
# PATH = '../model/faster-RCNN_FPN.pth'
height, width = 480,640
im = cv2.imread("../docs/input.jpg")
x = torch.tensor(np.reshape(im,(3,im.shape[0],im.shape[1])))
# x = torch.randn(3,height,width).cuda()
input = [{'image':x}]
model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
# model = torch.load(PATH)
cfg = get_cfg()
from detectron2.structures import ImageList
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

pred = DefaultPredictor(cfg)
out =pred(im)
# model = build_model(cfg)
# model.eval()

# out = model(input)

print(out)