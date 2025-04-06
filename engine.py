# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import box_iou, box_cxcywh_to_xyxy


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0, logger=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if isinstance(batch, dict):
            # Handle dictionary format
            image, text, target = batch['img'], batch['text'], batch['target']
        else:
            # Handle tuple format (img_data, text_data, target)
            image, text, target = batch
        
        image, text, target = image.to(device), text.to(device), target.to(device)
        
        output = model(image, text)
        
        loss = F.mse_loss(output, target)
        
        # Calculate IoU for logging
        pred_box = output.detach()
        gt_box = target
        iou_result, _ = box_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(gt_box))
        diag_iou = torch.diag(iou_result)
        
        # Log metrics
        if logger is not None and i % args.log_interval == 0:
            logger.log_metrics({
                'loss': loss.item(),
                'iou': diag_iou.mean().item(),
                'lr': optimizer.param_groups[0]['lr']
            }, epoch * len(data_loader) + i, prefix='train/')
            
            # Skip visualization since we don't have image_original in tuple format
        
        loss_dict = {'loss': loss}
        losses = sum(loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced['loss'].item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device, logger=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_ious = []
    iou_thresholds = args.iou_thresholds if hasattr(args, 'iou_thresholds') else [0.3, 0.5, 0.7]
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if isinstance(batch, dict):
            # Handle dictionary format
            image, text, target = batch['img'], batch['text'], batch['target']
        else:
            # Handle tuple format (img_data, text_data, target)
            image, text, target = batch
        
        image, text, target = image.to(device), text.to(device), target.to(device)
        
        # compute output
        output = model(image, text)
        
        loss = F.mse_loss(output, target)
        
        # Calculate IoU
        pred_box = output
        gt_box = target
        iou_result, _ = box_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(gt_box))
        diag_iou = torch.diag(iou_result)
        all_ious.append(diag_iou)
        
        # Skip visualization since we don't have image_original in tuple format
        
        loss_dict = {'loss': loss}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced['loss'].item()
        
        metric_logger.update(loss=loss_value)
    
    # Calculate accuracy at different IoU thresholds
    all_ious = torch.cat(all_ious)
    metrics = {}
    metrics['loss'] = metric_logger.loss.global_avg
    metrics['mean_iou'] = all_ious.mean().item()
    metrics['accu'] = (all_ious > 0.5).float().mean().item()  # Default accuracy at IoU=0.5
    
    for thresh in iou_thresholds:
        metrics[f'accuracy@{thresh}'] = (all_ious > thresh).float().mean().item()
    
    if logger is not None:
        logger.log_metrics(metrics, 0, prefix='val/')
        logger.log_histogram(all_ious.cpu().numpy(), 0, 'val/iou_distribution')
    
    # Print results
    print('-' * 50)
    print(f"Validation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print('-' * 50)
    
    return metrics


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy
        