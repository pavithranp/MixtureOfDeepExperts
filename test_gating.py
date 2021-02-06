import cv2
import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from network.GatingNetwork import GatingNetwork


def image_process(path1, path2, cfg):
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    image1 = cv2.imread(path1)
    image1 = cv2.resize(image1, (1920, 1080))
    height, width = image1.shape[:2]
    # image1 = aug.get_transform(image1).apply_image(image1)
    # image1 = torch.as_tensor(image1.astype("float32").transpose(2, 0, 1))

    image2 = cv2.imread(path2)
    image2 = cv2.resize(image1, (1920, 1080))
    height, width = image2.shape[:2]
    # image2 = aug.get_transform(image2).apply_image(image2)
    # image2 = torch.as_tensor(image2.astype("float32").transpose(2, 0, 1))
    image1 = torch.as_tensor(image1.transpose(2, 0, 1), dtype=torch.float32)
    image2 = torch.as_tensor(image2.transpose(2, 0, 1), dtype=torch.float32)
    return [{"rgb_image": image1, "depth_image": image2, "height": height, "width": width}]


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
print("InOutDoorDepth_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg2 = cfg.clone()
cfg.MODEL.WEIGHTS = "output/model_0024999.pth"
cfg2.MODEL.WEIGHTS = "output/model_0019999.pth"
rgb = image_process('docs/imagehd.png', 'docs/depthJet.png', cfg)
with torch.no_grad():
    gn = GatingNetwork(cfg, cfg2)

    outputs = gn(rgb)[0]
    im = cv2.imread('docs/imagehd.png')
    image1 = cv2.resize(im, (1920, 1080))

    v = Visualizer(image1,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    print(outputs)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('test', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
