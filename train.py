import os
from helper_methods import load_detectron_trainer_continue, load_samples_from_folder, convert_samples_to_detectron_dicts

# Configuration
dataset_dir = 'dataset'
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
trained_model_output_dir = 'training_output'
dataset_name = 'license-plate-detection-dataset'
class_labels = ["plate"]
prediction_score_threshold = 0.9
base_lr = 0.0025
max_iter = 1000
batch_size = 64

# Build dataset - load samples and filter out the ones with empty boxes
samples = load_samples_from_folder(dataset_dir)
samples = [s for s in samples if len(s.boxes) != 0]

# Build dataset - convert to detectron format and provide its function
detectron_dicts = convert_samples_to_detectron_dicts(samples)
def dataset_function():
    return detectron_dicts

# Build detectron trainer
cfg, trainer = load_detectron_trainer_continue(model_zoo_config_name, trained_model_output_dir, dataset_name, dataset_function, class_labels, prediction_score_threshold, base_lr, max_iter, batch_size)

# Make sure that output model dir exists and launch training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer.resume_or_load(resume=False)
trainer.train()
