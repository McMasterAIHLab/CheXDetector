import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from src.paths import path_outputs_dir

# Determining the no good data
def find_files_with_condition(data_file_paths, condition_value=-10, save_path=None):
    """
    Finds files where all rows match a specific condition (default is all values are -10)
    and optionally saves these file paths to a specified file.

    Parameters:
    - data_file_paths: List of paths to the data files.
    - condition_value: The value to check for the condition. Default is -10.
    - save_path: Path to save the list of file paths that meet the condition. If None, no file is saved.

    Returns:
    - List of file paths that meet the condition.
    """
    files_meeting_condition = []

    for file_path in data_file_paths:
        # Load the data
        data = np.load(file_path)
        
        # Check if all values in the data meet the condition
        if (data == condition_value).all():
            # If the condition is met, add the file path to the list
            files_meeting_condition.append(file_path)

    # Print the number of files found
    print(f"Found {len(files_meeting_condition)} files where all rows are {condition_value}.")

    # Optionally, save these file paths for exclusion
    if save_path:
        with open(save_path, "w") as f:
            for path in files_meeting_condition:
                f.write(f"{path}\n")
        print(f"Exclusion list saved to {save_path}")

    return files_meeting_condition

def compute_counter(trn_cls_label_file_paths):
    """
    Compute counter for classification labels. To be used to compute class weights.

    Args:
        trn_cls_label_file_paths (list or np.array): List of paths to the training classification label files.

    Returns:
        tensor: The computed counter tensor.
    """
    counter = torch.zeros([9])

    for file_path in trn_cls_label_file_paths:
        label = torch.from_numpy(np.load(file_path))
        summed = torch.sum(label, dim=0)
        counter += summed
        
    return counter
         
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

def clean_dataset(data_file_paths, cls_label_file_paths):
    """
    Finds files where all rows match a specific condition (default is all values are -10)
    and removes them from the dataset.

    Parameters:
    - data_file_paths: List of paths to the data files.
    - cls_label_file_paths: List of paths to the classification label files.

    Returns:
    - data_file_paths: List of paths to the filtered data files.
    - cls_label_file_paths: List of paths to the filtered classification label files.
    """
    # Now we want to delete this no good data from our entire dataset 

    # Find files_with_all_negative_ten
    files_with_all_negative_ten = find_files_with_condition(data_file_paths, condition_value=-10)

    # Filter out the data files and their corresponding labels
    filtered_data_file_paths = [path for path in data_file_paths if path not in files_with_all_negative_ten]
    filtered_cls_label_file_paths = [path for path, data_path in zip(cls_label_file_paths, data_file_paths) if data_path not in files_with_all_negative_ten]

    # Update the original lists if necessary
    data_file_paths = filtered_data_file_paths
    cls_label_file_paths = filtered_cls_label_file_paths

    return data_file_paths, cls_label_file_paths

def plot_model_roc_curves_with_label_names(model, dataloader, device, label_names, file_name = 'roc_curve.png'):
    """
    Computes and plots ROC curves and AUC scores for each label in a multi-label classification problem,
    using given label names. Saves plot to file_name in outputs folder.

    Parameters:
    - model: The trained model.
    - dataloader: DataLoader for the dataset to evaluate.
    - device: The device on which the model and data are or should be.
    - label_names: List of label names corresponding to each label.
    - file_name: '__.png' name of file to save ROC plot to
    """
    # Check if the number of label names matches the number of labels (10 in this case)
    if len(label_names) != 10:
        raise ValueError("The length of label_names must be 10.")

    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # Apply sigmoid to convert outputs to probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batch outputs
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute ROC AUC for each label and plot the ROC curve
    plt.figure(figsize=(10, 8))
    for i in range(all_labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Use label names in the legend
        plt.plot(fpr, tpr, label=f'{label_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Label')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path_outputs_dir, file_name))
