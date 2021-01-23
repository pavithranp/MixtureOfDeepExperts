from network.GatingNetwork import  GatingNetwork
from network.GatingNetwork import FRCNN_OutputLayer,FRCNN_ROIHeads
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch,cv2
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

def image_process(path,cfg):
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    image = cv2.imread(path)
    height, width = image.shape[:2]
    image = aug.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    return [{"image": image, "height": height, "width": width}]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
print("InOutDoorDepth_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

# cfg.MODEL.WEIGHTS = "output_RGB/model_final.pth"

model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
input = image_process('docs/depthJet.png',cfg)
model.eval()
with torch.no_grad():
#     pred = model(input)[0]

    RGB_network = FRCNN_ROIHeads(cfg)
    # # cfg.MODEL.WEIGHTS = "output/model_final.pth"
    # Depth_network = FRCNN_ROIHeads(cfg)
    x = RGB_network(input)
    # gn = GatingNetwork(cfg,cfg)
    # depth = image_process('docs/input.jpg',cfg)
    # rgb = image_process('docs/input.jpg',cfg)
    # gn(depth,rgb)


