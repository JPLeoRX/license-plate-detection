from typing import List, Dict
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from helper_objects import Sample
from utils import generate_uuid, save_image_pil_in_temp


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

        # Mask polygons
        triangle_1 = [
            x + w / 2, y + h / 2,
            x, y,
            x + w, y
        ]
        triangle_2 = [
            x + w / 2, y + h / 2,
            x + w, y,
            x + w, y + h
        ]
        triangle_3 = [
            x + w / 2, y + h / 2,
            x + w, y + h,
            x, y + h
        ]
        triangle_4 = [
            x + w / 2, y + h / 2,
            x, y + h,
            x, y
        ]

        bbox = [x, y, w, h] # (list[float], required): list of 4 numbers representing the bounding box of the instance.
        bbox_mode = bbox_mode # (int, required): the format of bbox. It must be a member of structures.BoxMode. Currently supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.
        category_id = labels_to_id_map[box.label] # (int, required): an integer in the range [0, num_categories-1] representing the category label. The value num_categories is reserved to represent the “background” category, if applicable.
        segmentation = [triangle_1, triangle_2, triangle_3, triangle_4]


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


def register_detectron_dataset(dataset_name: str, class_labels: List[str], dataset_function):
    DatasetCatalog.register(dataset_name, dataset_function)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
