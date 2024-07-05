import sys
sys.path.append('/home/shared/kailin/github/')

from src.paths import path_last_hidden_state, path_classification_labels, path_localized_disease_detection_model, path_localized_disease_detection_model_saved
from src.constants import disease_labels
from src.localized_disease_detection.utils import clean_dataset, compute_class_weights, compute_counter, plot_model_roc_curves_with_label_names
from src.localized_disease_detection.disease_classification_dataset import DiseaseClassificationDataset
from src.localized_disease_detection.mlp_enhanced_no_dropout import MLP_enhanced_no_dropout

import os

import glob
import torch
from torch import nn
from torch.utils.data import DataLoader

# Constants
EPOCHS = 15
BATCH_SIZE = 256
INPUT_DIM = 256
HIDDEN_DIM = 256
OUTPUT_DIM = 9
NUM_LAYERS = 7
LR = 5e-4
WEIGHT_DECAY = 1e-5
EVALUATE_EVERY_N_STEPS = 5

def get_dataloaders():
    trn_data_dir = os.path.join(path_last_hidden_state, "train")
    vld_data_dir  = os.path.join(path_last_hidden_state, "validation")
    tst_data_dir  = os.path.join(path_last_hidden_state, "test")

    trn_cls_label_dir = os.path.join(path_classification_labels, "train")
    vld_cls_label_dir = os.path.join(path_classification_labels, "validation")
    tst_cls_label_dir = os.path.join(path_classification_labels, "test")

    # Pattern to match files of interest
    trn_data_pattern = trn_data_dir + '/' + '*.npy'
    vld_data_pattern = vld_data_dir + '/' + '*.npy'
    tst_data_pattern = tst_data_dir + '/' + '*.npy' 

    trn_data_file_paths = glob.glob(trn_data_pattern, recursive=False)
    vld_data_file_paths = glob.glob(vld_data_pattern, recursive=False)
    tst_data_file_paths = glob.glob(tst_data_pattern, recursive=False)

    trn_cls_label_file_paths = [(trn_cls_label_dir + '/' + file_name.split('/')[-1]) for file_name in trn_data_file_paths]
    vld_cls_label_file_paths = [(vld_cls_label_dir + '/' + file_name.split('/')[-1]) for file_name in vld_data_file_paths]
    tst_cls_label_file_paths = [(tst_cls_label_dir + '/' + file_name.split('/')[-1]) for file_name in tst_data_file_paths]

    # Now we want to delete this no good data from our entire dataset 
    trn_data_file_paths, trn_cls_label_file_paths = clean_dataset(trn_data_file_paths, trn_cls_label_file_paths)
    vld_data_file_paths, vld_cls_label_file_paths = clean_dataset(vld_data_file_paths, vld_cls_label_file_paths)
    tst_data_file_paths, tst_cls_label_file_paths = clean_dataset(tst_data_file_paths, tst_cls_label_file_paths)

    trn_dataset = DiseaseClassificationDataset(trn_data_file_paths, trn_cls_label_file_paths)
    vld_dataset = DiseaseClassificationDataset(vld_data_file_paths, vld_cls_label_file_paths)
    tst_dataset = DiseaseClassificationDataset(tst_data_file_paths, tst_cls_label_file_paths)

    trn_dataloader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
    vld_dataloader = DataLoader(vld_dataset, batch_size=BATCH_SIZE, shuffle=False)
    tst_dataloader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return trn_cls_label_file_paths, trn_dataloader, vld_dataloader, tst_dataloader


def train_disease_model(device):
    trn_cls_label_file_paths, trn_dataloader, vld_dataloader, _ = get_dataloaders()

    # Instantiate the model
    model_no_dropout_7layers = MLP_enhanced_no_dropout(input_dim= INPUT_DIM, hidden_dim= HIDDEN_DIM, output_dim= OUTPUT_DIM, num_layers= NUM_LAYERS)
    model_no_dropout_7layers.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model_no_dropout_7layers.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Define the criterion
    counter = compute_counter(trn_cls_label_file_paths)
    weights = compute_class_weights(counter.numpy(), normalize=True)
    pos_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Initial validation loss
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(EPOCHS):
        model_no_dropout_7layers.train()
        running_loss = 0.0
        for i, data in enumerate(trn_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model_no_dropout_7layers(inputs)
            loss = criterion(outputs, labels)
            
            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 1:  
                print(f'Epoch [{epoch+1}/{EPOCHS}], Iteration [{i}/{len(trn_dataloader)}], Loss: {loss.item():.4f}')
                
        # Validation
        if (epoch+1) % EVALUATE_EVERY_N_STEPS == 0:  
            model_no_dropout_7layers.eval()
            with torch.no_grad():
                val_loss = 0.0
                for j, vld_data in enumerate(vld_dataloader):
                    vld_inputs, vld_labels = vld_data
                    vld_inputs, vld_labels = vld_inputs.to(device), vld_labels.to(device)
                    
                    # Forward pass
                    vld_outputs = model_no_dropout_7layers(vld_inputs)
                    vld_loss = criterion(vld_outputs, vld_labels)
                    
                    # Accumulate validation loss
                    val_loss += vld_loss.item()  # Fixed: should accumulate vld_loss.item() not vld_loss += vld_loss.item()
                
                avg_val_loss = val_loss / len(vld_dataloader)
                print(f'\nValidation Loss: {avg_val_loss:.4f}\n')
                
                # Save the best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model_no_dropout_7layers.state_dict(), path_localized_disease_detection_model)
                    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

def eval_disease_model(device):
    _ , _ , vld_dataloader, tst_dataloader = get_dataloaders()
    # Instantiate the model
    best_model_no_dropout_7layers = get_model(device)

    # Plot ROC curves
    plot_model_roc_curves_with_label_names(best_model_no_dropout_7layers, vld_dataloader, device, disease_labels, 'eval_disease_vld.png')
    plot_model_roc_curves_with_label_names(best_model_no_dropout_7layers, tst_dataloader, device, disease_labels, 'eval_disease_tst.png')

def get_model(device):
    # Instantiate the model
    best_model_no_dropout_7layers = MLP_enhanced_no_dropout(input_dim= INPUT_DIM, hidden_dim= HIDDEN_DIM, output_dim= OUTPUT_DIM, num_layers= NUM_LAYERS)
    best_model_no_dropout_7layers.load_state_dict(torch.load(path_localized_disease_detection_model_saved))
    best_model_no_dropout_7layers.to(device)
    return best_model_no_dropout_7layers

def get_localized_diseases(feature_layer, device, threshold = 0.5):
    model = get_model(device)
    model.eval()  # Set model to evaluation mode
    
    # Convert lists to tensors
    data = torch.tensor(feature_layer, dtype=torch.float32)

    with torch.no_grad():
        inputs = data.to(device)
        outputs = model(inputs)
        # Apply sigmoid to convert outputs to probabilities
        probs = torch.sigmoid(outputs)
        return probs.cpu().numpy() > threshold