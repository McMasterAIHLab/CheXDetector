from constants import anatomical_regions, progression_labels, diseases
from paths import path_outputs_dir
from representation_learning.representation_learning import get_valid_feature_layer_one, get_valid_feature_layer_two
from localized_disease_detection.localized_disease_detection import get_localized_diseases
from localized_disease_progression.localized_disease_progression import get_progression
import torch
import pandas as pd
import os

import sys
sys.path.append('/home/shared/kailin/github/')

from src.paths import path_imagenome_coco
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def infer_one_image(img_path):
    feature_layer_valid, valid_layers = get_valid_feature_layer_one(device, img_path)
    localized_diseases = get_localized_diseases(feature_layer_valid, device)
    return (structured_report(localized_diseases, valid_layers))

def infer_two_images(new_img_path, old_img_path):
    new_feature_layer_valid, valid_layers = get_valid_feature_layer_one(device, new_img_path)
    localized_diseases = get_localized_diseases(new_feature_layer_valid, device)

    new_feature_layer_prog, old_feature_layer_prog, valid_layers_prog = get_valid_feature_layer_two(device, new_img_path, old_img_path)
        
    prog = get_progression(device, new_feature_layer_prog, old_feature_layer_prog)
    return (structured_report(localized_diseases, valid_layers, prog, valid_layers_prog))

def get_localized_diseases_texts(localized_diseases, valid_regions):
    '''Returns list of text with localized diseases for table'''
    localized_diseases_texts = []

    ld_index = 0
    for i, _ in enumerate(anatomical_regions):
        if valid_regions[i]: # only 
            l = []
            for j, disease in enumerate(diseases):
                if (localized_diseases[ld_index][j]):
                    l.append(disease)
            if (l):
                localized_diseases_texts.append(', '.join(l))
            else:
                localized_diseases_texts.append(None)
            ld_index += 1
        else:
            localized_diseases_texts.append("N/A")

    return (localized_diseases_texts)

def get_progression_texts(progression_preds, valid_regions):
    '''Returns list of text with localized diseases for table'''
    if progression_preds:
        # progression_texts = [progression_labels[i] for i in progression_preds]
        prog_index = 0
        progression_texts = []
        for i, _ in enumerate(anatomical_regions):
            if valid_regions[i]: 
                progression_texts.append(progression_labels[progression_preds[prog_index]])
                prog_index += 1
            else:
                progression_texts.append("N/A")
    else:
        progression_texts = "N/A"
    return progression_texts


def structured_report(localized_diseases, valid_layers, prog=None, valid_layers_prog=None):
    report_data = {
        'Anatomical Region': anatomical_regions,
        'Findings': get_localized_diseases_texts(localized_diseases, valid_layers),
        'Progression': get_progression_texts(prog, valid_layers_prog)
    }
    df = pd.DataFrame(report_data)
    # Set 'Region' as the index
    df.set_index('Anatomical Region', inplace=True)
    return df

def main():

    test_image_file_name1 = 'fff3d520-503c91eb-bedb5882-8c38016c-10617b06'
    test_image_file_name2 = '1d3ad828-b83029af-e1f4eecd-b6b9d212-c099cb07'
    image_path1 = f'{path_imagenome_coco}/train/images/{test_image_file_name1}.jpg'
    image_path2 = f'{path_imagenome_coco}/train/images/{test_image_file_name2}.jpg'

    # If you want to infere on only one image, you can use "infer_one_image" function

    # table_df = infer_one_image(image_path1)
    table_df = infer_two_images(image_path1, image_path2)

    table_df.to_csv(os.path.join(path_outputs_dir, 'results.csv'))

if __name__ == '__main__':
    main()