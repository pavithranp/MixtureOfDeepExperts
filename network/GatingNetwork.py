# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.modeling import build_model
# import random
import torch
PATH = '../model/faster-RCNN.pth'
x = torch.randn(1,3,480,640)
input = [{'image':x}]
model = torch.load(PATH)
features = model.backbone(x.cuda())
# proposals, _ = model.proposal_generator(input, features)
# instances, _ = model.roi_heads(input, features, proposals)
print(features)