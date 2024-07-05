from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import os
import sys
from torch import nn
sys.path.append('/home/shared/kailin/github/')

from src.paths import path_last_hidden_state

def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners

# inspired by https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258
def post_process_object_detection(
    outputs, last_hidden_state = [], threshold: float = 0.5, target_sizes = None
    # self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
):
    """
    Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
    bottom_right_x, bottom_right_y) format. Only supports PyTorch.

    Args:
        outputs ([`DetrObjectDetectionOutput`]):
            Raw outputs of the model.
        threshold (`float`, *optional*):
            Score threshold to keep object detection predictions.
        target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
            `(height, width)` of each image in the batch. If unset, predictions will not be resized.
    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
        in the batch as predicted by the model.
    """
    out_logits, out_bbox = outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # Convert to [x0, y0, x1, y1] format
    boxes = _center_to_corners_format_torch(out_bbox)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, list):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]


    results = []

    for s, l, b, ls in zip(scores, labels, boxes, last_hidden_state):
    # for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        state = ls[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box, "last_hidden_states": state})
        # results.append({"scores": score, "labels": label, "boxes": box})

    return results

def preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    encoding = image_processor(images=image, return_tensors="pt")
    return {
        'pixel_values': encoding["pixel_values"].squeeze(),
        'pixel_mask': encoding['pixel_mask']
    }

def get_feature_layer(model, image, image_processor, device):
    model.eval()
    pixel_values = image["pixel_values"].unsqueeze(0).to(device)  # Add batch dimension
    pixel_mask = image["pixel_mask"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.tensor([[pixel_values.shape[2], pixel_values.shape[3]]]).to(device)
    results = post_process_object_detection(outputs, last_hidden_state=outputs['last_hidden_state'], target_sizes=orig_target_sizes, threshold=0.85)
    result = results[0]
    
    ideal = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    labels = result['labels'].cpu().numpy()
    last_hidden_states = result['last_hidden_states'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    doc = dict(zip(unique, counts))
    
    final_last_hidden_states = np.zeros((12, 256))

    for indx, label_num in enumerate(ideal):
        if label_num not in doc.keys():
            final_last_hidden_states[indx] = -10
        elif doc[label_num] != 1:
            dd = np.where(labels == label_num)[0]
            _scores = scores[dd]
            _argmax = np.argmax(_scores)
            max_score_index = dd[_argmax]
            final_last_hidden_states[indx] = last_hidden_states[max_score_index]
        else:
            final_last_hidden_states[indx] = last_hidden_states[np.where(labels == label_num)[0][0]]
    return final_last_hidden_states

def get_valid_feature_layer(feature_layer):
    '''Returns feature_layer with rows of -10 removed & array of boolean indicating which rows are valid'''
    valid_rows = ~(feature_layer == -10).all(axis=1)
    filtered_data = feature_layer[valid_rows]
    return filtered_data, valid_rows

def get_valid_feature_layer_2_imgs(new_feature_layer, old_feature_layer):
    '''Returns feature_layer with rows of -10 removed & array of boolean indicating which rows are valid'''
    new_valid_rows = ~(new_feature_layer == -10).all(axis=1)
    old_valid_rows = ~(old_feature_layer == -10).all(axis=1)
    valid_rows = new_valid_rows & old_valid_rows 
    new_filtered_data = new_feature_layer[valid_rows]
    old_filtered_data = old_feature_layer[valid_rows]
    return new_filtered_data, old_filtered_data, valid_rows

def save_feature_layers(model, device, data_loader, dataset, split_name):
    """
	Use the representation learning module to save the extracted feature layer for all images in the dataset
    to a specific directory corresponding to train, test, or validate.

    Args:
        model: representation learning model
        device
        image_processor
        data_loader: Dataloader corresponding to train, validate, or test.
        dataset: Dataset corresponding to train, validate, or test.
        split_name: train, validate, or test. (Used to save to correct directory)

    Returns:
        Nothing
    """
    
    dataset_dir = os.path.join(path_last_hidden_state, split_name)
    os.mkdir(dataset_dir)
    if(not os.path.isdir(dataset_dir)):
        raise Exception(f"Cannot find the directory '{dataset_dir}'. Make sure the dataset directory exists in the base repo.")
    
    for idx, batch in enumerate(tqdm(data_loader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = post_process_object_detection(outputs, last_hidden_state=outputs['last_hidden_state'], target_sizes=orig_target_sizes, threshold=0.85)

        ideal = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        for nn, result in enumerate(results):
            file_name = dataset.coco.imgs.get(batch['image_ids'][nn][0].item())['file_name']
            labels = result['labels'].cpu().numpy()
            last_hidden_states = result['last_hidden_states'].cpu().numpy()
            scores = result['scores'].cpu().numpy()

            unique, counts = np.unique(labels, return_counts=True)
            doc = dict(zip(unique, counts))
            
            final_last_hidden_states = np.zeros((12, 256))

            for indx, label_num in enumerate(ideal):
                if label_num not in doc.keys():
                    final_last_hidden_states[indx] = -10
                elif doc[label_num] != 1:
                    dd = np.where(labels == label_num)[0]
                    _scores = scores[dd]
                    _argmax = np.argmax(_scores)
                    max_score_index = dd[_argmax]
                    final_last_hidden_states[indx] = last_hidden_states[max_score_index]
                else:
                    final_last_hidden_states[indx] = last_hidden_states[np.where(labels == label_num)[0][0]]
            np.save(os.path.join(dataset_dir, f'{file_name.split(".")[0]}.npy'), final_last_hidden_states)
        
