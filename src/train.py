from representation_learning.representation_learning import train_repr_model, save_all_feature_layers
from localized_disease_detection.localized_disease_detection import train_disease_model
from localized_disease_progression.localized_disease_progression import train_progression_model

import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
This is the main file to train the model.
First, ONLY run 'train_repr_model' function to train DETR.
After the training is finished, set the path to the best checkpoint of the model in the variable 'path_representation_learning_model_dir_saved' in paths.py.
Now you can run the rest of the functions to save the features of DETR, train disease classification of progression detection module.
Keep in mind that you should run save_all_feature_layers prior to the training of the other two modules.
'''

print('Training DETR...')
train_repr_model()

# print('Saving last hidden states...')
# save_all_feature_layers(device)

# print('Training disease classification model...')
# train_disease_model(device)

# print('Training disease progression monitoring model...')
# train_progression_model(device)