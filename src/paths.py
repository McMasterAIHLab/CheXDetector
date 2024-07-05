"""
Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
MIMIC-CXR-JPG dataset paths should have a (sub-)directory called "files" in their directories.

path_imagenome_coco specifies the directory where the dataset for representation learning module will be created.
At the end, you'll see three folders created (train, validation, test), each containing an 'images' folder which contains the images and a JSON file which holds the coco format of the dataset.
You will also find the tensorflow format of the dataset in each split directory. This gives a good intuition about the data. 

path_last_hidden_state specifies the path that the last hidden states of DETR will be saved and used to train the downstream tasks (disease detection and progression monitoring).
It will contain three directories corresponding to three dataset splits. Each .npy file corresponds to one CXR and contains 12 rows (12 anatomical regions) and 256 columns (feature vector of that region).

path_classification_labels specifies the path in which classification labels will be saved. We will be having again the three folders for splits.
Each .npy file is mapped to one CXR and is a matrix of dimension [12, 9] corresponding to 12 anatomical regions and 9 findings (1 indicates the existence of the finding in the region).

path_progression_pairs specifies the directory whose list of CXRs used for progression detection is saved.
Each name corresponds to one CXR pair. The first part of the name (before '_') specifies the new CXR and the second part specifies the old CXR taken from the same patient.

path_progression_labels specifies the path in which all the progression labels will be saved. Each .txt file corresponds to one CXR pairs and it contains 12 rows.
Each row specifies whether the patient's condition in that anatomical region has improved, worsend or has not changed.

path_representation_learning_model_dir specifies the folder which checkpoints of fine-tuning DETR will be saved.

path_*_model specifies the paths which trained models will be saved. Training these models is relatively fast.

path_*_saved specify the paths to the best model checkpoints. These paths will be used for evaluation.

path_detr_evaluation specifies the path in which the detections and groundtruths will be saved in the format required by https://github.com/rafaelpadilla/Object-Detection-Metrics.
This will be used for evaluating the DETR model.

path_outputs_dir specifies the path in which the plots related to evaluation of disease detection and progression monitoring models 
and the result of the inference will be saved.
"""

path_chest_imagenome = "/home/shared/rgrg/physionet.org/files/chest-imagenome/1.0.0"
path_mimic_cxr_jpg = "/home/shared/CXR-multi-task/dataset/mimic-cxr-jpg/2.0.0"

path_imagenome_coco = '/home/shared/kailin/github/src/dataset/detr'

path_last_hidden_state = '/home/shared/kailin/github/src/dataset/last_hidden_state'
path_classification_labels = '/home/shared/kailin/github/src/dataset/classification_labels'
path_progression_pairs = '/home/shared/kailin/github/src/dataset/progression_labels/list'
path_progression_labels = '/home/shared/kailin/github/src/dataset/progression_labels/labels'

# Paths to save models to
path_representation_learning_model_dir = '/home/shared/kailin/github/src/checkpoints/detr'
path_localized_disease_detection_model = '/home/shared/kailin/github/src/checkpoints/localized_disease_detection_model.pth'
path_localized_disease_progression_model = '/home/shared/kailin/github/src/checkpoints/localized_disease_progression_model.pth'

# Paths of best models
path_representation_learning_model_dir_saved = '/home/shared/kailin/github/src/checkpoints/detr/1/epoch=0-step=2562.ckpt'
path_localized_disease_detection_model_saved = '/home/shared/kailin/github/src/checkpoints/localized_disease_detection_model.pth'
path_localized_disease_progression_model_saved = '/home/shared/kailin/github/src/checkpoints/localized_disease_progression_model.pth'


# Path to save DETR detections for evaluation
path_detr_evaluation = '/home/shared/kailin/github/src/dataset/evaluate'

# Paths outputs
path_outputs_dir = '/home/shared/kailin/github/outputs'