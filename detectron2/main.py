"""
source
https://www.youtube.com/watch?v=GoItxr16ae8
"""
import detectron2
import sys

sys.path.insert(1, "J:/detectron2/projects/PointRend")
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.projects import point_rend


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


if __name__ == "__main__":
    # register_coco_instances("my_dataset_train", {}, "E:/robo_ukrainiec/datasets/coco2017/annotations/instances_train2017.json", "E:/robo_ukrainiec/datasets/coco2017/train2017")
    # register_coco_instances("my_dataset_val", {}, "E:/robo_ukrainiec/datasets/coco2017/annotations/instances_val2017.json", "E:/robo_ukrainiec/datasets/coco2017/val2017")
    register_coco_instances(
        "my_dataset_train",
        {},
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001/train_kjn_parse_dataset_002.json",
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001/image",
    )
    register_coco_instances(
        "my_dataset_val",
        {},
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001/valid_kjn_parse_dataset_002.json",
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001/image",
    )
    # my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    # dataset_dicts = DatasetCatalog.get("my_dataset_train")

    # visulazitaion
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("image", vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)

    cfg = get_cfg()
    # point_rend.add_pointrend_config(cfg)
    # cfg.merge_from_file(
    #     "J:/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    # )
    # cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    # cfg.MODEL.WEIGHTS = "detectron_output/model_0004999.pth"

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 0.0001

    # cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = (
        150000  # adjust up if val mAP is still rising, adjust down if overfit
    )
    # cfg.SOLVER.GAMMA = 0.05

    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = "./detectron_output"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    # cfg.MODEL.POINT_HEAD.NUM_CLASSES = 9
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    print(cfg)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
