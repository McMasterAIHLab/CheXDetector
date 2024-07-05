import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import path_outputs_dir

from src.localized_disease_progression.progression_attention_dataset import ProgressionAttentionDataset

def load_words_from_txt(file_path):
    """
    Loads words from a .txt file, where each line contains one word,
    and stores them in a list.

    Parameters:
    - file_path: The path to the .txt file.

    Returns:
    - A list of words from the file.
    """
    with open(file_path, 'r') as file:
        # Read lines and strip newline characters
        words = [line.strip() for line in file.readlines()]
    return words


def count_specific_instances(data_and_label_paths):
    """
    Counts the occurrences of 'no change', 'worsened', and 'improved' across
    all .txt files specified in the data_and_label_paths list.

    Parameters:
    - data_and_label_paths: A list of tuples, where each tuple contains paths
      to data files and the third element is the path to a .txt file.

    Returns:
    - A dictionary with counts for 'no change', 'worsened', and 'improved'.
    """
    # Initialize counts
    counts = {'no change': 0, 'worsened': 0, 'improved': 0}
    
    # Iterate over each path and process the .txt files
    for _, _, txt_file_path in data_and_label_paths:
        words_list = load_words_from_txt(txt_file_path)
        # Increment count for each occurrence
        for word in words_list:
            if word in counts:
                counts[word] += 1

    return counts

def categorize_data_by_labels(data_and_label_paths):
    """
    Categorizes data into 'improved', 'worsened', and 'no change' based on labels in txt files.

    Parameters:
    - data_and_label_paths: A list of tuples, where each tuple contains paths to two arrays and a .txt file.

    Returns:
    - A dictionary with keys 'improved', 'worsened', and 'no change', each containing a list of tuples
      in the format (arr_1_path, arr_2_path, idx_of_row).
    """
    # Initialize lists for each category
    categories = {'improved': [], 'worsened': [], 'no change': []}

    for arr_1_path, arr_2_path, label_path in data_and_label_paths:
        # Load the labels from the txt file
        labels = load_words_from_txt(label_path)
        
        new = np.load(arr_1_path, allow_pickle=True)
        old = np.load(arr_2_path, allow_pickle=True)
        
        assert new.shape[0] == len(labels), f"Labels and data do not match in {arr_1_path}"
        assert old.shape[0] == len(labels), f"Labels and data do not match in {arr_2_path}" 

        # Iterate through labels and categorize
        for idx, label in enumerate(labels):
            if label in categories:
                categories[label].append((arr_1_path, arr_2_path, idx))
    
    return categories


# Helper function to load words from a .txt file
def load_words_from_txt(file_path):
    """
    Loads words from a .txt file, where each line contains one word,
    and stores them in a list.

    Parameters:
    - file_path: The path to the .txt file.

    Returns:
    - A list of words from the file.
    """
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words


def compute_class_weights(class_occurrences, normalize=True, round_decimals=2):
    """
    Compute class weights for handling class imbalance in multi-label classification,
    with options for normalization and rounding.

    Args:
        class_occurrences (list or np.array): The number of occurrences for each class.
        normalize (bool): Whether to normalize weights so that the minimum weight is 1.0.
        round_decimals (int): The number of decimal places to round the weights.

    Returns:
        np.array: The computed and optionally normalized and rounded weights for each class.
    """
    # Convert class_occurrences to a numpy array if it's not already
    class_occurrences = np.array(class_occurrences, dtype=np.float32)
    
    # Compute inverse frequencies
    inverse_frequencies = 1.0 / class_occurrences
    
    if normalize:
        # Normalize weights so that the smallest weight is 1.0
        weights = inverse_frequencies / np.min(inverse_frequencies)
    else:
        weights = inverse_frequencies
    
    # Round the weights to the specified number of decimal places
    rounded_weights = np.round(weights, decimals=round_decimals)
    
    return rounded_weights

# Functions to evaluate model performance

def calculate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_loaded in dataloader:
            data, labels = data_loaded
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    confusion = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1_score, confusion



def calculate_metrics_weighted(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_loaded in dataloader:
            data, labels = data_loaded
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    confusion = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1_score, confusion



def evaluate_and_plot_metrics(model, dataloader, device, class_names, file_name = 'confusion_matrix.png'):
    """
    Computes and plots confusion matrix and accuracy, precision, recall and F1 scores for each label 
    in a multi-class classification problem, using given class names. Saves plot to file_name in outputs folder.

    Parameters:
    - model: The trained model.
    - dataloader: DataLoader for the dataset to evaluate.
    - device: The device on which the model and data are or should be.
    - class_names: List of class names corresponding to each class.
    - file_name: '__.png' name of file to save plot to
    """
    # Calculate metrics
    accuracy, precision, recall, f1_score, confusion = calculate_metrics(model, dataloader, device)

    # Plotting
    plt.figure(figsize=(10, 7))

    # Plot confusion matrix
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Add text with metrics
    plt.subplots_adjust(bottom=0.1)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.15, f'Precision: {precision:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.2, f'Recall: {recall:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.25, f'F1-Score: {f1_score:.2f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.savefig(os.path.join(path_outputs_dir, file_name))



def clean_data(categorized_data):
    """
    Removes instances with invalid rows (entire row is -10) in the new and old image arrays.

    Args:
        categorized_data (dict): Dictionary containing categorized data lists under 'improved', 'worsened', and 'no change'.

    Returns:
        dict: Cleaned categorized data with invalid instances removed.
    """
    # Initialize a new dictionary to hold the cleaned data
    cleaned_data = {category: [] for category in categorized_data.keys()}

    for category, data_list in categorized_data.items():
        for new_img_path, old_img_path, idx in data_list:
            # Load the new and old images
            new_img = np.load(new_img_path)
            old_img = np.load(old_img_path)

            # Check if the specified rows in new and old images are valid
            if not np.all(new_img[idx, :] == -10) and not np.all(old_img[idx, :] == -10):
                cleaned_data[category].append((new_img_path, old_img_path, idx))

    return cleaned_data

def progression_pair_names(pairs_txt_file_path):
    """
    Determines tuples of progression pair names for localized disease progression.

    Args:
        pairs_txt_file_path (str): Path to txt file with pairs

    Returns:
        list of tuples: Tuples of progression pair names.
    """

    # Use the with statement to open the file and ensure it gets closed properly
    with open(pairs_txt_file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

    # Optional: Remove newline characters from each line
    progression_pairs_names = [line.strip() for line in lines]
    progression_pairs_names = [((file_path.split('.'))[0]).split('_') for file_path in progression_pairs_names]

    return progression_pairs_names

def get_dataset(pairs_txt_file_path, data_dir, progression_labels_dir):
    """
    Gets dataset for localized disease progression.

    Args:
        pairs_txt_file_path (str): Path to txt file with pairs
        data_dir (str): Path to directory with data
        progression_labels_dir (str): Path to directory with labels

    Returns:
        Dataset: dataset.
    """
    progression_pairs_names = progression_pair_names(pairs_txt_file_path)
    data_and_label_paths = [(data_dir + '/' + new + '.npy', data_dir + '/' + old + '.npy', progression_labels_dir + '/' + new + '_' + old + '.txt') for new,old in progression_pairs_names]

    categorized_data = categorize_data_by_labels(data_and_label_paths)
    categorized_data = clean_data(categorized_data)

    improved_list = categorized_data['improved']
    worsened_list = categorized_data['worsened']
    no_change_list = categorized_data['no change']

    combined_data_tuples =  [((x[0], x[1], x[2]), 0) for x in improved_list] + \
                            [((x[0], x[1], x[2]), 1) for x in no_change_list] + \
                            [((x[0], x[1], x[2]), 2) for x in worsened_list] 

    # Create the dataset
    return ProgressionAttentionDataset(combined_data_tuples)
