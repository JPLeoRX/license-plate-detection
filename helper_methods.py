import os
import tempfile
from typing import List, Dict
import concurrent.futures
from itertools import repeat
import uuid
import cv2
import numpy
from PIL.Image import Image
from PIL import Image as image_main
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from helper_objects import Sample, Prediction, Rectangle, LabeledBox


def generate_uuid() -> str:
    return str(uuid.uuid4())

# Image related methods
#-----------------------------------------------------------------------------------------------------------------------
def open_image_pil(image_path: str) -> Image:
    return image_main.open(image_path)

def save_image_pil(pil_image: Image, image_path: str) -> str:
    pil_image.save(image_path)
    return image_path

def save_image_pil_in_temp(pil_image: Image, image_name: str) -> str:
    folder_path = str(tempfile.gettempdir())
    image_path = folder_path + '/' + image_name + '.' + pil_image.format.lower()
    return save_image_pil(pil_image, image_path)

def convert_image_pil_to_cv(pil_image: Image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

def view_image_cv(cv_image):
    cv2.namedWindow('Viewing image', cv2.WINDOW_NORMAL)
    cv2.imshow('Viewing image', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def view_image_pil(pil_image: Image):
    view_image_cv(convert_image_pil_to_cv(pil_image))
#-----------------------------------------------------------------------------------------------------------------------



# Pascal VOC dataset
#-----------------------------------------------------------------------------------------------------------------------
def load_sample_from_png_and_pascal_voc_xml(image_file_path: str, xml_file_path: str) -> Sample:
    image_pil = open_image_pil(image_file_path)
    xml_file = open(xml_file_path, 'r')
    xml_text = xml_file.read()
    xml_file.close()

    name = [line for line in xml_text.split('\n') if '<filename>' in line][0].replace('<filename>', '').replace('</filename>', '').strip()
    boxes = []

    objects = xml_text.split('<object>')
    objects = objects[1:]
    for object in objects:
        lines = object.split('\n')
        line_name = [line for line in lines if '<name>' in line][0]
        line_xmin = [line for line in lines if '<xmin>' in line][0]
        line_ymin = [line for line in lines if '<ymin>' in line][0]
        line_xmax = [line for line in lines if '<xmax>' in line][0]
        line_ymax = [line for line in lines if '<ymax>' in line][0]

        label = line_name.replace('<name>', '').replace('</name>', '').strip()
        xmin = int(line_xmin.replace('<xmin>', '').replace('</xmin>', '').strip())
        ymin = int(line_ymin.replace('<ymin>', '').replace('</ymin>', '').strip())
        xmax = int(line_xmax.replace('<xmax>', '').replace('</xmax>', '').strip())
        ymax = int(line_ymax.replace('<ymax>', '').replace('</ymax>', '').strip())

        x = xmin / image_pil.width
        y = ymin / image_pil.height
        w = (xmax - xmin) / image_pil.width
        h = (ymax - ymin) / image_pil.height

        region = Rectangle(x, y, w, h)
        box = LabeledBox(label, region)
        boxes.append(box)

    return Sample(name, image_pil, boxes)


def load_sample_from_folder(image_and_xml_file_name: str, folder_path: str) -> Sample:
    # Build image file path, trying different image format options
    image_file_path = folder_path + '/' + image_and_xml_file_name + '.png'
    if not os.path.isfile(image_file_path):
        image_file_path = image_file_path.replace('.png', '.jpeg')
        if not os.path.isfile(image_file_path):
            image_file_path = image_file_path.replace('.jpeg', '.jpg')

    # Build XML file path, and show warning if no markup found
    xml_file_path = folder_path + '/' + image_and_xml_file_name + '.xml'
    if not os.path.isfile(xml_file_path):
        print('load_sample_from_folder(): Warning! XML not found, xml_file_path=' + str(xml_file_path))
        return None

    # Load sample
    return load_sample_from_png_and_pascal_voc_xml(image_file_path, xml_file_path)


def load_samples_from_folder(folder_path: str) -> List[Sample]:
    samples = []

    # Get all files, strip their extensions and resort
    all_files = os.listdir(folder_path)
    all_files = ['.'.join(f.split('.')[:-1]) for f in all_files]
    all_files = set(all_files)
    all_files = sorted(all_files)

    # Load samples in parallel
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    for sample in executor.map(load_sample_from_folder, all_files, repeat(folder_path)):
        if sample is not None:
            samples.append(sample)

    # Filter out None values
    samples = [s for s in samples if s is not None]

    return samples
#-----------------------------------------------------------------------------------------------------------------------



# Detectron dataset
#-----------------------------------------------------------------------------------------------------------------------
def build_labels_maps(samples: List[Sample]) -> (Dict[str, int], Dict[int, str]):
    labels = []
    for sample in samples:
        for box in sample.boxes:
            if box.label not in labels:
                labels.append(box.label)
    labels = sorted(labels)
    labels_to_id_map = {}
    id_to_labels_map = {}
    for i in range(0, len(labels)):
        labels_to_id_map[labels[i]] = i
        id_to_labels_map[i] = labels[i]
    return labels_to_id_map, id_to_labels_map

def convert_sample_to_detectron_dict(sample: Sample, labels_to_id_map: Dict[str, int], bbox_mode: BoxMode) -> Dict:
    # Generate ID, load image and save it in temp
    id = generate_uuid()
    image_pil = sample.image
    image_path = save_image_pil_in_temp(image_pil, id)

    # Build common
    file_name = image_path # the full path to the image file.
    height = image_pil.height # integer. The shape of the image.
    width = image_pil.width # integer. The shape of the image.
    image_id = id # (str or int): a unique id that identifies this image. Required by many evaluators to identify the images, but a dataset may use it for different purposes.
    annotations = [] # (list[dict]): Required by instance detection/segmentation or keypoint detection tasks. Each dict corresponds to annotations of one instance in this image, and may contain the following keys:

    # Build boxes
    for box in sample.boxes:
        x = int(box.region.x * width)
        y = int(box.region.y * height)
        w = int(box.region.w * width)
        h = int(box.region.h * height)
        polygon = [x, y, x + w, y, x, y + h, x + w, y + h]
        bbox = [x, y, w, h] # (list[float], required): list of 4 numbers representing the bounding box of the instance.
        bbox_mode = bbox_mode # (int, required): the format of bbox. It must be a member of structures.BoxMode. Currently supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.
        category_id = labels_to_id_map[box.label] # (int, required): an integer in the range [0, num_categories-1] representing the category label. The value num_categories is reserved to represent the “background” category, if applicable.
        segmentation = [polygon]


        annotation = {
            'bbox': bbox,
            'bbox_mode': bbox_mode,
            'category_id': category_id,
            'segmentation': segmentation
        }

        annotations.append(annotation)

    return {
        'file_name': file_name,
        'height': height,
        'width': width,
        'image_id': image_id,
        'annotations': annotations
    }

def convert_samples_to_detectron_dicts(samples: List[Sample]) -> List[Dict]:
    labels_to_id_map, id_to_labels_map = build_labels_maps(samples)
    detectron_dicts = []
    for sample in samples:
        d = convert_sample_to_detectron_dict(sample, labels_to_id_map, BoxMode.XYWH_ABS)
        detectron_dicts.append(d)
    return detectron_dicts
#-----------------------------------------------------------------------------------------------------------------------



# Detectron training & prediction
#-----------------------------------------------------------------------------------------------------------------------
def _empty_dataset_function():
    return []


def load_detectron_predictor(model_zoo_config_name: str, trained_model_output_dir: str, dataset_name: str, class_labels: List[str], prediction_score_threshold: float) -> (CfgNode, DefaultPredictor):
    DatasetCatalog.register(dataset_name, _empty_dataset_function)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = trained_model_output_dir
    cfg.MODEL.WEIGHTS = trained_model_output_dir + "/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    return cfg, DefaultPredictor(cfg)


def load_detectron_trainer_continue(model_zoo_config_name: str, trained_model_output_dir: str, dataset_name: str, dataset_function, class_labels: List[str], prediction_score_threshold: float, base_lr: float, max_iter: int, batch_size: int) -> (CfgNode, DefaultTrainer):
    DatasetCatalog.register(dataset_name, dataset_function)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = trained_model_output_dir
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = trained_model_output_dir + "/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    return cfg, DefaultTrainer(cfg)


def load_detectron_trainer_initial(model_zoo_config_name: str, trained_model_output_dir: str, dataset_name: str, dataset_function, class_labels: List[str], prediction_score_threshold: float, base_lr: float, max_iter: int, batch_size: int) -> (CfgNode, DefaultTrainer):
    DatasetCatalog.register(dataset_name, dataset_function)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = trained_model_output_dir
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    return cfg, DefaultTrainer(cfg)


def visualize_detectron_outputs(cfg, image_cv, outputs):
    v = Visualizer(image_cv[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    debug_image_cv = out.get_image()[:, :, ::-1]
    view_image_cv(debug_image_cv)


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
#-----------------------------------------------------------------------------------------------------------------------