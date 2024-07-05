import sys
sys.path.append('/home/shared/kailin/github/')

from src.paths import path_last_hidden_state, path_progression_labels, path_progression_pairs, path_localized_disease_progression_model, path_localized_disease_progression_model_saved
from src.localized_disease_progression.utils import get_dataset, evaluate_and_plot_metrics
from src.localized_disease_progression.mlp_enhanced import MLP_enhanced
from src.localized_disease_progression.preprocess_progression import preprocess_progression
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

# Constants
EPOCHS = 10
BATCH_SIZE = 256
INPUT_DIM = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2
CLS_OUTPUT_DIM = 3
DROPOUT_RATE = 0
LR = 1e-3
WEIGHT_DECAY = 1e-5
EVALUATE_EVERY_N_STEPS = 5

def get_dataloaders():
    # processing data
    trn_data_dir = os.path.join(path_last_hidden_state, "train")
    vld_data_dir  = os.path.join(path_last_hidden_state, "validation")
    tst_data_dir  = os.path.join(path_last_hidden_state, "test")

    progression_labels_dir = path_progression_labels

    # Define the path to your text file
    trn_pairs_txt_file_path = os.path.join(path_progression_pairs, "train.txt")
    vld_pairs_txt_file_path  = os.path.join(path_progression_pairs, "val.txt")
    tst_pairs_txt_file_path  = os.path.join(path_progression_pairs, "test.txt")

    trn_dataset = get_dataset(trn_pairs_txt_file_path, trn_data_dir, progression_labels_dir)
    trn_dataloader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)

    vld_dataset = get_dataset(vld_pairs_txt_file_path, vld_data_dir, progression_labels_dir)
    vld_dataloader = DataLoader(vld_dataset, batch_size=BATCH_SIZE, shuffle=False)

    tst_dataset = get_dataset(tst_pairs_txt_file_path, tst_data_dir, progression_labels_dir)
    tst_dataloader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return trn_dataloader, vld_dataloader, tst_dataloader

def train_progression_model(device):
    trn_dataloader, vld_dataloader, _ = get_dataloaders()

    # Instantiate the model
    model_2layers = MLP_enhanced(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, num_layers = NUM_LAYERS, output_dim = CLS_OUTPUT_DIM, dropout_rate = DROPOUT_RATE, apply_to_output = False)
    model_2layers.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model_2layers.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Initial validation loss
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(EPOCHS):
        model_2layers.train()
        running_loss = 0.0

        for i, (data, labels) in enumerate(trn_dataloader):
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model_2layers(data)
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
            model_2layers.eval()
            val_loss = 0.0
            with torch.no_grad():
                for j, (vld_data, vld_labels) in enumerate(vld_dataloader):
                    vld_data, vld_labels = vld_data.to(device), vld_labels.to(device)
                    
                    # Forward pass
                    vld_outputs = model_2layers(vld_data)
                    vld_loss = criterion(vld_outputs, vld_labels)
                    
                    # Accumulate validation loss
                    val_loss += vld_loss.item()
                
                avg_val_loss = val_loss / len(vld_dataloader)
                print(f'\nValidation Loss: {avg_val_loss:.4f}\n')
                
                # Save the best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model_2layers.state_dict(), path_localized_disease_progression_model)
                    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

def load_model(device):
    # Load best model
    best_model_2layers = MLP_enhanced(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, num_layers = NUM_LAYERS, output_dim = CLS_OUTPUT_DIM, dropout_rate = DROPOUT_RATE, apply_to_output = False)
    best_model_2layers.load_state_dict(torch.load(path_localized_disease_progression_model_saved))
    best_model_2layers.to(device)
    return best_model_2layers

def eval_progression_model(device):
    _, vld_dataloader, tst_dataloader = get_dataloaders()
    best_model_2layers = load_model(device)
    

    # Model Evaluation
    class_names = ['Improved', 'No Change', 'Worsened']
    evaluate_and_plot_metrics(best_model_2layers, vld_dataloader, device, class_names, 'eval_progression_vld.png')
    evaluate_and_plot_metrics(best_model_2layers, tst_dataloader, device, class_names, 'eval_progression_tst.png')

def get_progression(device, new_img, old_img):
    model = load_model(device)

    row_tensors = []
    for idx in range(len(new_img)):
        row_tensors.append(preprocess_progression(new_img, old_img, idx).unsqueeze(0))
    
    return get_progression_rows(model, row_tensors, device)

def get_progression_rows(model, row_tensors, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in row_tensors:
            data = data.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    return all_preds