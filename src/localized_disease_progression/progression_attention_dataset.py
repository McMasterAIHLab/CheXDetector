import numpy as np
import torch
from torch.utils.data import Dataset

class ProgressionAttentionDataset(Dataset):
    def __init__(self, data_tuples, transform=None):
        self.data_tuples = data_tuples
        self.transform = transform

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        (new_img_path, old_img_path, row_idx), label = self.data_tuples[idx]
        
        # Load images
        new_img = np.load(new_img_path)
        old_img = np.load(old_img_path)

        # Compute the absolute difference between the feature vectors of the new and old images
        diff_img = new_img - old_img
        abs_diff = np.abs(diff_img)
        
        # Store the row of interest as a tensor
        diff_tensor = torch.from_numpy(diff_img[row_idx]).float()
        # print(abs_diff_tensor.shape)

        # Compute attention weights for all regions (12x12)
        attention_weights = self.compute_attention_for_all_regions(abs_diff) 
        
        # Use attention_weights to compute the weighted sum of the feature vectors
        output_values_for_each_region = np.matmul(attention_weights, abs_diff) # (12, 256) array
        # Now we average over the dim=0 to get the weighted sum of the feature vectors
        averaged_row = np.mean(output_values_for_each_region, axis=0) # (256,) array
        # print(averaged_row.shape)
        
        # Basically, what is happening here is that we define only 1 global attention vector and use that as 
        # auxiliary information (along with the feature vector for each region) to get the prediction for each row
        
        averaged_row_tensor = torch.from_numpy(averaged_row).float()
        # print(weighted_row_sum_tensor.shape)
        
        # Apply transform if any
        if self.transform:
            averaged_row_tensor = self.transform(averaged_row_tensor)
            diff_tensor = self.transform(diff_tensor)
        
        label_tensor = torch.tensor(label).long()

        return torch.cat((diff_tensor, averaged_row_tensor), dim=0), label_tensor

    @staticmethod
    def compute_attention_for_all_regions(feature_array):
        sqrt_dk = np.sqrt(feature_array.shape[1])
        query = feature_array
        keys = feature_array
        attention_weights = scaled_dot_product_attention(query, keys, sqrt_dk)
        return attention_weights # (12, 12) array


def scaled_dot_product_attention(query, keys, sqrt_dk):
    """
    Compute the scaled dot product attention.
    
    Args:
    query (numpy.ndarray): Query vector (feature vector of the specified region).
    keys (numpy.ndarray): Key vectors (feature vectors of all regions).
    sqrt_dk (float): Square root of the dimension of the key vector.

    Returns:
    numpy.ndarray: The attention weights.
    """
    # Compute dot product of query and keys (transpose of keys)
    matmul_qk = np.dot(query, keys.T)

    # Scale matmul_qk
    scaled_attention_logits = matmul_qk / sqrt_dk

    # Apply softmax to get probabilities (attention weights)
    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    return attention_weights