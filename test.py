import os
import time
#from helper_methods import load_detectron_predictor, open_image_pil, convert_image_pil_to_cv, convert_detectron_outputs_to_predictions, visualize_detectron_outputs
from utils import open_image_pil
from utils_dataset_pascalvoc import load_samples_from_folder
from utils_dataset_detectron import convert_samples_to_detectron_dicts, register_detectron_dataset
from utils_model import build_config, build_predictor, run_prediction

# Configuration
dataset_dir = 'dataset'
dataset_test_dir = 'dataset_test'
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
trained_model_output_dir = 'training_output'
dataset_name = 'license-plate-detection-dataset'
class_labels = ["plate"]
prediction_score_threshold = 0.9
base_lr = 0
max_iter = 0
batch_size = 0

# Build dataset - load samples and filter out the ones with empty boxes
samples = load_samples_from_folder(dataset_dir)
samples = [s for s in samples if len(s.boxes) != 0]

# Build dataset - convert to detectron format and provide its function and then register it
detectron_dicts = convert_samples_to_detectron_dicts(samples)
def dataset_function():
    return detectron_dicts
register_detectron_dataset(dataset_name, class_labels, dataset_function)

# Build detectron predictor
cfg = build_config(model_zoo_config_name, dataset_name, class_labels, trained_model_output_dir, prediction_score_threshold, base_lr, max_iter, batch_size)
predictor = build_predictor(cfg)

# Test images from training dataset
image_paths = [
    # dataset_dir + '/1280px-1988_Honda_Accord_Aerodeck_2.0_EX_(15243754588).jpeg',
    # dataset_dir + '/1280px-Alpina_B12_(E38)_(16531947875).jpeg',
    # dataset_dir + '/1280px-INFINITI_FX50.jpeg',
    # dataset_dir + '/1280px-Lexus_GS_300_(44110082832).jpeg',
    # dataset_dir + '/audi-a8-i-d2-restyling-1998-2002-sedan-exterior-4.jpeg',
    # dataset_dir + '/BMW_E38_oxford2.jpeg',
    # dataset_dir + '/Toyota_Century_Front_side_Shinjuku.jpeg',
    # dataset_dir + '/1280px-Osaka_Auto_Messe_2019_(28)_-_LB-Works_Hakosuka.jpeg',
    # dataset_dir + '/1280px-Полиция_Kia_Ceed.jpeg',
    # dataset_dir + '/1280px-1970-1_Porsche_911_T_(11031423685).jpeg',
    # dataset_dir + '/1280px-1988_Porsche_911_in_Dark_Blue_-_Rear.jpeg',
    # dataset_dir + '/1280px-Honda_S2000_(31030043298).jpeg',
    # dataset_dir + '/1280px-Mazda_Eunos_Roadster_(11531604964).jpeg',
    # dataset_dir + '/1280px-Mercedes-Benz_Atego_1324.jpeg',
    # dataset_dir + '/1280px-Porsche_911_Carrera_3.2_Coupé-_P6280053.jpeg',
    # dataset_dir + '/1280px-Porsche_911_G-Serie_Wien_29_July_2020_JM.jpeg',
    # dataset_dir + '/1280px-Porsche_Porsche_cayenne_turbo_(6905976777).jpeg',
    # dataset_dir + '/1024px-Porsche_Cayenne_Turbo.jpeg',
    # dataset_dir + '/1280px-Mazda_Eunos_Roadster_V-special_Green.jpeg',
    # dataset_dir + '/1280px-Porsche_Cayenne_with_SIR_license_plate_of_Queensland_01.jpeg',
    # dataset_dir + '/1993_Ford_F-150.jpeg',
    dataset_dir + '/Photo 11.08.2021 05 16 06.png',
    dataset_dir + '/Photo 12.08.2021 03 22 07.png',
]

# Dataset test images
dataset_test_image_paths = os.listdir(dataset_test_dir)
dataset_test_image_paths = sorted(dataset_test_image_paths)
dataset_test_image_paths = [p for p in dataset_test_image_paths if p.startswith('Photo')]
dataset_test_image_paths = [dataset_test_dir + '/' + p for p in dataset_test_image_paths]

# Merge test & train images
image_paths.extend(dataset_test_image_paths)

# Run predictions
for image_path in image_paths:
    image_pil = open_image_pil(image_path)
    run_prediction(cfg, predictor, class_labels, image_pil)
