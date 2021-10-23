import os
import time
from typing import List
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from PIL.Image import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from helper_objects import Prediction, Rectangle
from utils import convert_pil_to_cv, debug_image_cv


def build_config(
        model_zoo_config_name: str,
        dataset_name: str, class_labels: List[str],
        trained_model_output_dir: str,
        prediction_score_threshold: float,
        base_lr: float, max_iter: int, batch_size: int
) -> CfgNode:
    trained_model_weights_path = trained_model_output_dir + "/model_final.pth"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = trained_model_output_dir
    cfg.DATALOADER.NUM_WORKERS = 8
    if os.path.exists(trained_model_weights_path):
        cfg.MODEL.WEIGHTS = trained_model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg


def run_training(cfg: CfgNode):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def build_predictor(cfg: CfgNode) -> DefaultPredictor:
    return DefaultPredictor(cfg)


def visualize_detectron_outputs(cfg: CfgNode, image_cv, outputs):
    v = Visualizer(image_cv[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image_cv = out.get_image()[:, :, ::-1]
    debug_image_cv(output_image_cv)


def convert_detectron_outputs_to_predictions(class_labels: List[str], outputs) -> List[Prediction]:
    results = []
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    for i in range(0, len(pred_boxes)):
        box = pred_boxes[i].tensor.numpy()[0]
        score = float(scores[i].numpy())
        label_key = int(pred_classes[i].numpy())
        label = class_labels[label_key]

        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        region = Rectangle(int(x), int(y), int(w), int(h))

        prediction = Prediction(label, score, region)
        results.append(prediction)

    return results


def run_prediction(cfg: CfgNode, predictor: DefaultPredictor, class_labels: List[str], pil_image: Image):
    # Prep image
    cv_image = convert_pil_to_cv(pil_image)

    # Run prediction and time it
    t1 = time.time()
    outputs = predictor(cv_image)
    t2 = time.time()
    d = t2 - t1

    # Debug predictions
    visualize_detectron_outputs(cfg, cv_image, outputs)
    predictions = convert_detectron_outputs_to_predictions(class_labels, outputs)
    print('run_prediction(): Testing "' + pil_image.filename + '" took ' + str(round(d, 2)) + ' seconds, and resulted in predictions ' + str(predictions))
