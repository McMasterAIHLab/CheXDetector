import numpy as np
import torch
from torch.utils.data import Dataset

class DiseaseClassificationDataset(Dataset):
    def __init__(self, data_file_paths, cls_label_file_paths):
        super(DiseaseClassificationDataset, self).__init__()
        self.data = []
        self.labels = []
        
        # Load all data and labels, treating each row as an instance
        for data_path, label_path in zip(data_file_paths, cls_label_file_paths):
            data = np.load(data_path)  # Load data
            labels = np.load(label_path)  # Load labels
            
            # Check if any row is all -10, and if so, exclude those rows
            valid_rows = ~(data == -10).all(axis=1)
            
            # Filter data and labels to include only valid rows
            filtered_data = data[valid_rows]
            filtered_labels = labels[valid_rows]
            
            # Extend the dataset with the filtered rows
            self.data.extend(filtered_data)
            self.labels.extend(filtered_labels)
        
        # Convert lists to tensors
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        
    def __len__(self):
        # The dataset length is the total number of rows (instances)
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return the data and label for the specified index
        return self.data[idx], self.labels[idx]