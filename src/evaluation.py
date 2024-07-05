from representation_learning.representation_learning import eval_repr_model
from localized_disease_detection.localized_disease_detection import eval_disease_model
from localized_disease_progression.localized_disease_progression import eval_progression_model
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
This is the main file to evaluate the 3 models.
For evaluating representation learning model (DETR), we used https://github.com/rafaelpadilla/Object-Detection-Metrics.
'eval_repr_model' function creates 'detections' and 'groundtruth' directories which are used by this package.
Follow the instructions on the repo to perform evaluation.
'''

eval_repr_model(device)
eval_disease_model(device)
eval_progression_model(device)