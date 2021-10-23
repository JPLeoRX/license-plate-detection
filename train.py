from utils_dataset_pascalvoc import load_samples_from_folder
from utils_dataset_detectron import convert_samples_to_detectron_dicts, register_detectron_dataset
from utils_model import build_config, run_training

# Configuration
dataset_dir = 'dataset'
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
trained_model_output_dir = 'training_output'
dataset_name = 'license-plate-detection-dataset'
class_labels = ["plate"]
prediction_score_threshold = 0.9
base_lr = 0.0025
max_iter = 200
batch_size = 64

# Build dataset - load samples and filter out the ones with empty boxes
samples = load_samples_from_folder(dataset_dir)
samples = [s for s in samples if len(s.boxes) != 0]

# Build dataset - convert to detectron format and provide its function and then register it
detectron_dicts = convert_samples_to_detectron_dicts(samples)
def dataset_function():
    return detectron_dicts
register_detectron_dataset(dataset_name, class_labels, dataset_function)

# Build detectron config & run trainer
cfg = build_config(model_zoo_config_name, dataset_name, class_labels, trained_model_output_dir, prediction_score_threshold, base_lr, max_iter, batch_size)
run_training(cfg)
