import sys
sys.path.append('/home/shared/kailin/github/')

from src.paths import path_imagenome_coco, path_representation_learning_model_dir, path_representation_learning_model_dir_saved
from src.representation_learning.coco_detection import CocoDetection
from src.representation_learning.utils import preprocess_image, get_feature_layer, get_valid_feature_layer, get_valid_feature_layer_2_imgs, save_feature_layers
from src.representation_learning.save_all_cocos import save_all_cocos

import os

import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# File paths
dataset = path_imagenome_coco

TRAIN_DIRECTORY = os.path.join(dataset, "train", 'images')
VAL_DIRECTORY = os.path.join(dataset, "validation", 'images')
TEST_DIRECTORY = os.path.join(dataset, "test", 'images')

# Parameters
RUN = 2
BATCH_SIZE = 13
EPOCHS = 1
MODEL_CHECKPOINT = 'facebook/detr-resnet-50'
NUM_WORKERS = 23
LR = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_VAL =  0.1
ACCUMULATE_GRAD_BATCHES = 5
LOG_EVERY_N_STEPS = 50

# Image processor
image_processor = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=MODEL_CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs.logits

    def training_step(self, batch, batch_idx):
        loss, loss_dict, logits = self.common_step(batch, batch_idx)     
        self.log("training_loss", loss, batch_size=BATCH_SIZE, prog_bar=True, sync_dist=True)
        
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=BATCH_SIZE, prog_bar=True, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, labels = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss, batch_size=BATCH_SIZE ,prog_bar=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=BATCH_SIZE, prog_bar=True, sync_dist=True)
            
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    image_ids = [item[1]['image_id'] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels,
        'image_ids': image_ids
    }

# Create dataset and dataloader
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

def train_repr_model():
    model = Detr(lr=LR, lr_backbone=LR_BACKBONE, weight_decay=WEIGHT_DECAY)

    os.mkdir(os.path.join(path_representation_learning_model_dir, str(RUN)))

    logger = CSVLogger("logs", name='detr')
    checkpoint_callback = ModelCheckpoint(dirpath=f'{path_representation_learning_model_dir}/{RUN}', monitor='validation/loss', save_last=True, save_top_k=1)
    trainer = Trainer(devices=[1], accelerator="gpu", max_epochs=EPOCHS, gradient_clip_val=GRADIENT_CLIP_VAL, accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES, log_every_n_steps=LOG_EVERY_N_STEPS,
            callbacks=[checkpoint_callback, RichProgressBar()], logger=logger)

    trainer.fit(model)

def load_repr_model(device):
    model = Detr.load_from_checkpoint(path_representation_learning_model_dir_saved, map_location=device)
    model.to(device)
    return model

def get_valid_feature_layer_one(device, img_path):
    """
    Given an image path, returns the valid feature layer and a list of which features were deemed valid.

    Args:
        device
        img_path (str): Path to the CXR image.

    Returns:
        feature_layer_valid: Valid feature layer.
        valid_layers: Boolean list of layers which are valid.
    """
    model = load_repr_model(device)

    img = preprocess_image(img_path, image_processor)
    new_feature_layer = get_feature_layer(model, img, image_processor, device)
    return get_valid_feature_layer(new_feature_layer)

def get_valid_feature_layer_two(device, new_img_path, old_img_path):
    """
    Given 2 image paths, returns 2 valid feature layer and a list of which features were deemed valid

    Args:
        device
        new_img_path (str): Path to the newer CXR image.
        old_img_path (str): Path to the older CXR image.

    Returns:
        new_feature_layer_valid: Valid feature layer for new CXR.
        old_feature_layer_valid: Valid feature layer for old CXR.
        valid_layers: Boolean list of layers which are valid for both.
    """
    model = load_repr_model(device)

    new_img = preprocess_image(new_img_path, image_processor)
    old_img = preprocess_image(old_img_path, image_processor)

    new_feature_layer = get_feature_layer(model, new_img, image_processor, device)
    old_feature_layer = get_feature_layer(model, old_img, image_processor, device)

    return get_valid_feature_layer_2_imgs(new_feature_layer, old_feature_layer)

def save_all_feature_layers(device):
    """
	Use the representation learning module to save the extracted feature layer for all images (train, test, and validate).

    Args:
        device

    Returns:
        Nothing
    """

    model = load_repr_model(device)

    model.eval()
    save_feature_layers(model, device, VAL_DATALOADER, VAL_DATASET, 'validation')
    save_feature_layers(model, device, TEST_DATALOADER, TEST_DATASET, 'test')
    save_feature_layers(model, device, TRAIN_DATALOADER, TRAIN_DATASET, 'train')

    return

def eval_repr_model(device):
    model = load_repr_model(device)
    save_all_cocos(model, device, TEST_DATALOADER, TEST_DATASET)
