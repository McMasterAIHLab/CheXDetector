import numpy as np
import torch
from localized_disease_progression.progression_attention_dataset import scaled_dot_product_attention

def preprocess_progression(new_img, old_img, row_idx):
    '''Takes in npy feature extractions of 2 images with invalid regions (-10) already removed'''
    # Compute the absolute difference between the feature vectors of the new and old images
    diff_img = new_img - old_img
    abs_diff = np.abs(diff_img)
    
    # Store the row of interest as a tensor
    diff_tensor = torch.from_numpy(diff_img[row_idx]).float()

    # Compute attention weights for all regions (12x12)
    attention_weights = compute_attention_for_all_regions(abs_diff) 
        
    # Use attention_weights to compute the weighted sum of the feature vectors
    output_values_for_each_region = np.matmul(attention_weights, abs_diff) # (12, 256) array
    
    # Now we average over the dim=0 to get the weighted sum of the feature vectors
    averaged_row = np.mean(output_values_for_each_region, axis=0) # (256,) array
        
    # Basically, what is happening here is that we define only 1 global attention vector and use that as 
    # auxiliary information (along with the feature vector for each region) to get the prediction for each row
        
    averaged_row_tensor = torch.from_numpy(averaged_row).float()
    return torch.cat((diff_tensor, averaged_row_tensor), dim=0)

def compute_attention_for_all_regions(feature_array):
    sqrt_dk = np.sqrt(feature_array.shape[1])
    query = feature_array
    keys = feature_array
    attention_weights = scaled_dot_product_attention(query, keys, sqrt_dk)
    return attention_weights # (12, 12) array

