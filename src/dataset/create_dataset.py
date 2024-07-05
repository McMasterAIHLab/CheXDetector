import os
import sys
import pandas as pd
import numpy as np
import cv2
import json
import shutil

sys.path.append('/home/shared/kailin/github/')

from src.paths import path_chest_imagenome, path_mimic_cxr_jpg, path_imagenome_coco, path_classification_labels, path_progression_labels, path_progression_pairs
from tensorflow_to_coco import tensorflow_to_coco

anatomical_regions = [
    'right upper lung zone',
    'right mid lung zone',
    'right lower lung zone',
    'right costophrenic angle',
    'right hilar structures',
    'right apical zone',
    'left upper lung zone',
    'left mid lung zone',
    'left lower lung zone',
    'left costophrenic angle',
    'left hilar structures',
    'cardiac silhouette',
]

diseases = [
    'lung opacity',
    'pleural effusion',
    'atelectasis',
    'enlarged cardiac silhouette',
    'pulmonary edema/hazy opacity',
    'pneumothorax',
    'consolidation',
    'fluid overload/heart failure',
    'pneumonia'
]

def extract_annotations_progression():
    splits = ['train', 'test', 'val']
    scene_graphs_names = np.array([])
    for split in splits:
        array = np.loadtxt(os.path.join(path_progression_pairs, f"{split}.txt"), dtype='str')
        scene_graphs_names = np.append(scene_graphs_names, array)

    for idx, i in enumerate(scene_graphs_names):
        print(idx)
        comparisons_per_regions = np.full(shape=(12, 1), dtype=object, fill_value='no change')

        scene_graphs_name = f"{i.split('_')[0]}_SceneGraph.json"

        f = open(os.path.join(path_chest_imagenome, 'silver_dataset', 'scene_graph', scene_graphs_name), 'r')
        data = json.load(f)
        f.close()
        for relationship in data['relationships']:
            if relationship['bbox_name'] in anatomical_regions:
                _diseases = np.array(relationship['attributes'], dtype=object).flatten()
                _diseases = np.array(list(set(_diseases)))
                _final_diseases = []
                for j in _diseases:
                    j = j.split('|')
                    if j[1] == 'yes':
                        _final_diseases.append(j[2])
                
                if set(_final_diseases).isdisjoint(set(diseases)):
                    comparisons_per_regions[anatomical_regions.index(relationship['bbox_name'])] = 'disease conflict'

                comparisons = []
                for j in relationship['relationship_names']:
                    comparisons.append(j)
                comparisons = list(set(comparisons))
                
                if len(comparisons) == 1:                    
                    comparison = comparisons[0].split('|')[2]
                    comparisons_per_regions[anatomical_regions.index(relationship['bbox_name'])] = comparison
                else:
                    comparisons_per_regions[anatomical_regions.index(relationship['bbox_name'])] = 'conflict'
        save_name = i.split('.')[0]

        np.savetxt(f"{os.path.join(path_progression_labels, f'{save_name}.txt')}", comparisons_per_regions, fmt="%s")

def extract_annotations_disease(split):
    os.mkdir(os.path.join(path_classification_labels, split))
    scene_graphs_names = None
    
    df = pd.read_csv(os.path.join(path_chest_imagenome, 'silver_dataset', 'splits', f"{split}.csv"))
    df['dicom_id'] = df['dicom_id'].astype(str) + '_SceneGraph.json'
    scene_graphs_names = df['dicom_id'].tolist()

    df = pd.DataFrame({'filename': [], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'region': []})
    dictionary_list = []
    not_found = np.array([], dtype=object)

    for index, scene_graph in enumerate(scene_graphs_names):
        print(index)
        
        file_name = scene_graph.split('_')[0] + '.jpg'
        im = cv2.imread(os.path.join(path_imagenome_coco, split, 'images', file_name))
        height, width, channels = im.shape
        try:
            f = open(os.path.join(path_chest_imagenome, 'silver_dataset', 'scene_graph', scene_graph), 'r')
        except FileNotFoundError:
            not_found = np.append(not_found, scene_graph)
        else:
            data = json.load(f)
            f.close()

            for obj in data['objects']:
                if obj['bbox_name'] in anatomical_regions:
                    flag = False
                    findings = np.array([])
                    for attribute in data['attributes']:
                        if attribute['bbox_name'] == obj['bbox_name']:
                            for attr in attribute['attributes']:
                                findings = np.concatenate((findings, attr))

                    findings = np.array(list(set(findings)))
                    for finding in findings:
                        finding = finding.split('|')
                        if (finding[0] == 'anatomicalfinding' or finding[0] == 'disease') and finding[1] == 'yes' and finding[2] in diseases:
                            flag = True
                            dictionary_list.append({'filename': file_name, 'width': width, 'height': height, 'class': finding[2], 'xmin': obj['original_x1'], 'ymin': obj['original_y1'], 'xmax': obj['original_x2'], 'ymax':  obj['original_y2'], 'region': obj['bbox_name']})
                            
                    if not flag:
                        dictionary_list.append({'filename': file_name, 'width': width, 'height': height, 'class': 'no finding', 'xmin': obj['original_x1'], 'ymin': obj['original_y1'], 'xmax': obj['original_x2'], 'ymax':  obj['original_y2'], 'region': obj['bbox_name']})
    
    df = pd.DataFrame.from_dict(dictionary_list)
    unique_values = df['filename'].unique()

    i = 0
    for value in unique_values:
        print(i)
        result = np.zeros((12, 9))
        filtered_df = df[df['filename'] == value]
        filtered_df = filtered_df.reset_index()
        filename = filtered_df['filename'][0]

        flag = False
        for index, row in filtered_df.iterrows():
            label = row['class']
            anatomical_region = row['region']
            if label != 'no finding':
                flag = True
                result[anatomical_regions.index(anatomical_region)][diseases.index(label)] = 1
        np.save(os.path.join(path_classification_labels, split, f"{filename.split('.')[0]}.npy"), result.astype(int))
        i += 1

def extract_annotations_anatomical_regions(split):
    df = pd.read_csv(os.path.join(path_chest_imagenome, 'silver_dataset', 'splits', f"{split}.csv"))
    df['dicom_id'] = df['dicom_id'].astype(str) + '_SceneGraph.json'
    scene_graphs_names = df['dicom_id'].tolist()

    _annotations = pd.DataFrame({'filename': [], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []})
    dictionary_list = []
    not_found = np.array([], dtype=object)

    for index, scene_graph in enumerate(scene_graphs_names):
        print(index)
        
        file_name = scene_graph.split('_')[0] + '.jpg'

        im = cv2.imread(os.path.join(path_imagenome_coco, split, 'images', file_name))

        height, width, _ = im.shape

        try:
            f = open(os.path.join(path_chest_imagenome, 'silver_dataset', 'scene_graph', scene_graph), 'r')
        except:
            not_found = np.append(not_found, scene_graph)
        else:
            data = json.load(f)
            f.close()

            for obj in data['objects']:
                if obj['bbox_name'] in anatomical_regions:
                    dictionary_list.append({
                        'filename': file_name,
                        'width': width,
                        'height': height,
                        'class': obj['bbox_name'],
                        'xmin': obj['original_x1'],
                        'ymin': obj['original_y1'],
                        'xmax': obj['original_x2'],
                        'ymax':  obj['original_y2']
                    })

    _annotations = pd.DataFrame.from_dict(dictionary_list)
    _annotations.to_csv(os.path.join(path_imagenome_coco, split, '_annotations_anatomical_regions.csv'), index=False)
    tensorflow_to_coco(os.path.join(path_imagenome_coco, split, '_annotations_anatomical_regions.csv'),
                       os.path.join(path_imagenome_coco, split, 'images', '_annotations_anatomical_regions.json'))

def move_images_dataset(split):
    os.mkdir(os.path.join(path_imagenome_coco, split))
    os.mkdir(os.path.join(path_imagenome_coco, split, 'images'))

    df = pd.read_csv(os.path.join(path_chest_imagenome, 'silver_dataset', 'splits', f"{split}.csv"))
    df = df.reset_index()
    for index, row in df.iterrows():
        print(index)
        image_path = row['path'].split('.')[0] + '.jpg'
        shutil.copy(os.path.join(path_mimic_cxr_jpg, image_path), os.path.join(path_imagenome_coco, split, 'images'))

def main():
    splits = ['train', 'validation', 'test']

    for split in splits:
        move_images_dataset(split)
        extract_annotations_anatomical_regions(split)
        extract_annotations_disease(split)

    extract_annotations_progression()
if __name__ == '__main__':
    main()