import argparse
from collections import defaultdict
import json
import os
import random
from scipy.ndimage import distance_transform_edt
import torch
import numpy as np
from tqdm import tqdm
import gc
import cv2
from pycocotools import coco
from config import dataset_registry
from src.m2n2_model import M2N2SegmentationModel
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator


def click_center_of_error_region(pred_mask: np.ndarray,
                                 gt_mask: np.ndarray,
                                 unassigned_mask: np.ndarray):
    # Get the potential wrong area where either a FP or FN in all pixels which have segmentation
    wrong_mask = ((pred_mask != gt_mask) * (1 - unassigned_mask)).astype(int)
    if np.sum(wrong_mask) == 0:
        wrong_mask = (gt_mask * (1 - unassigned_mask)).astype(int)
    wrong_mask_padded = np.zeros((wrong_mask.shape[0] + 2, wrong_mask.shape[1] + 2), dtype=int)
    wrong_mask_padded[1:-1, 1:-1] = wrong_mask
    sd = distance_transform_edt(wrong_mask_padded)

    # Select point
    yx = np.unravel_index(np.argmax(sd), wrong_mask_padded.shape)
    point_position = (int(yx[1] - 1), int(yx[0] - 1))
    point_position = (max(0, min(gt_mask.shape[1] - 1, point_position[0])), max(0, min(gt_mask.shape[0] - 1, point_position[1])))
    return point_position, bool(gt_mask[point_position[1], point_position[0]] == 1)


def evaluate_pipeline_by_false_area_clicking(img: np.ndarray,
                                             gt_mask: np.ndarray,
                                             unassigned_mask: np.ndarray,
                                             model,
                                             max_num_point_prompts):

    masks = np.zeros((max_num_point_prompts + 1, img.shape[0], img.shape[1]), dtype=int)
    points = []
    points_in_segment = []
    for i in range(max_num_point_prompts):

        # Add point to list of points
        pos, in_seg = click_center_of_error_region(pred_mask=masks[i], gt_mask=gt_mask, unassigned_mask=unassigned_mask)
        points.append(pos)
        points_in_segment.append(in_seg)

        # Do the prediction
        model_out = model.segment(img, points, points_in_segment)
        # _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        # ax1.imshow(img)
        # ax1.imshow(model_out.cpu().numpy(), alpha=.5)
        # ax2.imshow(img)
        # ax2.imshow(gt_mask, alpha=.5)
        # plt.show()
        masks[i + 1] = model_out.cpu().numpy()
        
    return {
        'masks': masks[1:],
        'points': np.array(points, dtype=int),
        'points_in_segment': np.array(points_in_segment, dtype=bool)
    }


def get_metrics_from_coco_outs(per_num_clicks_evaluations):
    noc_counter_85 = defaultdict(lambda: 20)
    noc_counter_90 = defaultdict(lambda: 20)
    noc_counter_95 = defaultdict(lambda: 20)
    for j, evaluation in enumerate(per_num_clicks_evaluations):
        click = j + 1
        for ann in evaluation:
            if ann['metrics']['iou'] >= 0.85:
                noc_counter_85[ann['id']] = min(noc_counter_85[ann['id']], click)
            else:
                # This branch is needed so that 20 points entry gets created
                noc_counter_85[ann['id']] = noc_counter_85[ann['id']]
            if ann['metrics']['iou'] >= 0.90:
                noc_counter_90[ann['id']] = min(noc_counter_90[ann['id']], click)
            else:
                # This branch is needed so that 20 points entry gets created
                noc_counter_90[ann['id']] = noc_counter_90[ann['id']]
            if ann['metrics']['iou'] >= 0.95:
                noc_counter_95[ann['id']] = min(noc_counter_95[ann['id']], click)
            else:
                # This branch is needed so that 20 points entry gets created
                noc_counter_95[ann['id']] = noc_counter_95[ann['id']]
    noc_85 = float(np.mean(list(noc_counter_85.values())))
    noc_90 = float(np.mean(list(noc_counter_90.values())))
    noc_95 = float(np.mean(list(noc_counter_95.values())))
    metrics = {
        'NoC85': noc_85,
        'NoC90': noc_90,
        'NoC95': noc_95,
        'NoC85_per_ann_id': noc_counter_85,
        'NoC90_per_ann_id': noc_counter_90,
        'NoC95_per_ann_id': noc_counter_95,
    }

    for j, evaluation in enumerate(per_num_clicks_evaluations):
        ious_per_ann_id = {a['id']: a['metrics']['iou'] for a in evaluation}
        metrics[f'mIoU_{j + 1}pts'] = float(np.mean(list(ious_per_ann_id.values())))
        metrics[f'mIoU_{j + 1}pts_per_ann_id'] = ious_per_ann_id
    return metrics


def evaluate_model_on_coco_object_dataset(model,
                                          image_folder,
                                          annotation_fname,
                                          max_num_prompt_points=20,
                                          treat_unassigned_as_background=False,
                                          num_random_image_ids=-1,
                                          random_image_ids_seed=2024):
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize coco dataset and get image ids
    cocoGt = coco.COCO(annotation_fname)
    imgIds = cocoGt.getImgIds()

    if num_random_image_ids > 0:
        random.seed(random_image_ids_seed)
        random.shuffle(imgIds)
        imgIds = imgIds[:num_random_image_ids]

    per_num_clicks_evaluations = [[] for _ in range(max_num_prompt_points)]
    for img_idx, imgId in enumerate(pbar := tqdm(imgIds, f'Evaluating {model.__class__.__name__} on {annotation_fname}')):
        
        # -------------------------
        # Load Image and Annotation
        # -------------------------
        img_info = cocoGt.loadImgs([imgId])[0]
        img_fname = os.path.join(image_folder, img_info['file_name'])
        img = cv2.imread(img_fname)[:, :, ::-1]
        ann_ids = cocoGt.getAnnIds([imgId])
        annotations = cocoGt.loadAnns(ann_ids)

        # -----------------------------
        # Decode all segmentation masks
        # -----------------------------
        gt_masks = np.zeros((len(annotations), img.shape[0], img.shape[1]), dtype=int)
        for ann_idx, annotation in enumerate(annotations):
            gt_masks[ann_idx] = cocoGt.annToMask(annotation)
        assigned_mask = np.ones(gt_masks.shape[1:], dtype=int) if treat_unassigned_as_background else np.max(gt_masks, axis=0)
        unassigned_mask = 1 - assigned_mask
        
        # --------------------
        # Perform Segmentation
        # --------------------
        predictions = []
        for ann_idx, gt_mask in enumerate(gt_masks):
            pred = evaluate_pipeline_by_false_area_clicking(
                img=img,
                gt_mask=gt_mask,
                unassigned_mask=unassigned_mask,
                model=model,
                max_num_point_prompts=max_num_prompt_points,
            )
            predictions.append(pred)

        # --------
        # Evaluate
        # --------
        for ann_idx, gt_mask in enumerate(gt_masks):
            pred = predictions[ann_idx]
            annotation = annotations[ann_idx]
            annotation_id = annotation['id']
            for interaction_idx, pred_mask in enumerate(pred['masks']):
                cm = {
                    'TP': int(np.sum(np.logical_and(np.logical_and(pred_mask, gt_mask), assigned_mask))),
                    'FP': int(np.sum(np.logical_and(np.logical_and(pred_mask, np.logical_not(gt_mask)), assigned_mask))),
                    'FN': int(np.sum(np.logical_and(np.logical_and(np.logical_not(pred_mask), gt_mask), assigned_mask))),
                    'TN': int(np.sum(np.logical_and(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)), assigned_mask))),
                    'ALL': int(np.sum(assigned_mask)),
                }
                per_num_clicks_evaluations[interaction_idx].append({
                    'id': annotation_id,
                    'metrics': {
                        'iou': cm['TP'] / (cm['TP'] + cm['FP'] + cm['FN']),             # https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
                        'acc': (cm['TP'] + cm['TN']) / cm['ALL'],                       # https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall
                        'dice': 2 * cm['TP'] / (2 * cm['TP'] + cm['FP'] + cm['FN'])     # https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-accuracy
                    }
                })

        # ------------------
        # Update progressbar
        # ------------------
        current_metrics = get_metrics_from_coco_outs(per_num_clicks_evaluations)
        postfix_txt = [f'NoC85: {current_metrics["NoC85"]:.5f} NoC90: {current_metrics["NoC90"]:.5f} NoC95: {current_metrics["NoC95"]:.5f}; mIoU']
        for j in range(len(per_num_clicks_evaluations)):
            miou = current_metrics[f'mIoU_{j + 1}pts']
            postfix_txt.append(f'{j + 1}pt:{miou:.3f}')
        pbar.set_postfix_str(' '.join(postfix_txt))

    return get_metrics_from_coco_outs(per_num_clicks_evaluations)


def evaluate_dataset(dataset_key,
                     model,
                     max_num_prompt_points=20,
                     treat_unassigned_as_background=False,
                     out_metrics_fname=None,
                     num_random_image_ids=-1,
                     random_image_ids_seed=2024):
    image_folder, annotation_fname = dataset_registry[dataset_key]

    # ---------
    # Inference
    # ---------
    metrics = evaluate_model_on_coco_object_dataset(
        image_folder=image_folder,
        annotation_fname=annotation_fname,
        model=model,
        max_num_prompt_points=max_num_prompt_points,
        treat_unassigned_as_background=treat_unassigned_as_background,
        num_random_image_ids=num_random_image_ids,
        random_image_ids_seed=random_image_ids_seed
    )

    # -------------
    # Store results
    # -------------
    if out_metrics_fname is not None:
        os.makedirs(os.path.dirname(out_metrics_fname), exist_ok=True)
        with open(out_metrics_fname, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Dataset Evaluation")
    for key in dataset_registry.keys():
        parser.add_argument(f'--{key}', action='store_true', default=False)
    selected_datasets = vars(parser.parse_args())

    for dataset_key in dataset_registry.keys():
        if not selected_datasets[dataset_key]:
            continue

        attention_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')
        model = M2N2SegmentationModel(attention_aggregator)
        evaluate_dataset(
            dataset_key=dataset_key,
            model=model,
            max_num_prompt_points=20,
            treat_unassigned_as_background=True,
            out_metrics_fname=f'./out/{dataset_key}_metrics.json'
        )
        model.print_statistics()
        del attention_aggregator
        del model


if __name__ == '__main__':
    main()
