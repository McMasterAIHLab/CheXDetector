from tqdm import tqdm
import numpy as np
import torch
import sys
import os
sys.path.append('/home/shared/kailin/github/')
from src.constants import anatomical_regions
from src.paths import path_detr_evaluation
from src.representation_learning.utils import post_process_object_detection


def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners

def save_all_cocos(model, device, data_loader, dataset):
    """
    Feeds the (test) dataset to the representation learning model and saves the output with the corresponding label in coco format.
    Args:
        model: representation learning model
        device
        data_loader: Dataloader corresponding to test.
        dataset: Dataset corresponding to test.
    Returns:
        Nothing
    """

    dataset_dir = path_detr_evaluation
    if(not os.path.isdir(dataset_dir)):
        raise Exception(f"Cannot find the directory {dataset_dir}. Make sure the dataset directory exists in the base repo.")

    model.eval()

    for batch in tqdm(data_loader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

        if orig_target_sizes is not None:
            img_h = torch.Tensor([i[0] for i in orig_target_sizes])
            img_w = torch.Tensor([i[1] for i in orig_target_sizes])

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
        for jj, item in enumerate(labels):
            file_name = dataset.coco.imgs.get(batch['image_ids'][jj][0].item())['file_name']
            uu = _center_to_corners_format_torch(labels[jj]['boxes'])
            boxes_new = uu * scale_fct[jj]
            lines = []
            for ii in range(len(labels[jj]['class_labels'])):
                _str = f"{labels[jj]['class_labels'][ii]} {boxes_new[ii][0]} {boxes_new[ii][1]} {boxes_new[ii][2]} {boxes_new[ii][3]}"
                lines.append(_str)
            with open(f"{dataset_dir}/groundtruths/{file_name.split('.')[0]}.txt", 'w') as f:
                for line in lines:
                    f.write(f"{line}\n")
                f.close()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        results = post_process_object_detection(outputs, last_hidden_state=outputs['last_hidden_state'], target_sizes=orig_target_sizes, threshold=0.85)

        ideal = [i for i in range(1, len(anatomical_regions) + 1)][::-1]

        for nn, result in enumerate(results):
            lines = []
            file_name = dataset.coco.imgs.get(batch['image_ids'][nn][0].item())['file_name']
            labels = result['labels'].cpu().numpy()
            scores = result['scores'].cpu().numpy()

            unique, counts = np.unique(labels, return_counts=True)
            doc = dict(zip(unique, counts))

            for label_num in ideal:
                _str = None
                if label_num not in doc.keys():
                    continue
                elif doc[label_num] != 1:
                    dd = np.where(labels == label_num)[0]
                    _scores = scores[dd]
                    _argmax = np.argmax(_scores)
                    max_score_index = dd[_argmax]
                    _str = f"{result['labels'][max_score_index]} {result['scores'][max_score_index]} {result['boxes'][max_score_index][0]} {result['boxes'][max_score_index][1]} {result['boxes'][max_score_index][2]} {result['boxes'][max_score_index][3]}"
                    lines.append(_str)
                else:
                    dd = np.where(labels == label_num)[0][0]
                    _str = f"{result['labels'][dd]} {result['scores'][dd]} {result['boxes'][dd][0]} {result['boxes'][dd][1]} {result['boxes'][dd][2]} {result['boxes'][dd][3]}"
                    lines.append(_str)
                    
            with open(f"{dataset_dir}/detections/{file_name.split('.')[0]}.txt", 'w') as f:
                for line in lines:
                    f.write(f"{line}\n")
                f.close()
