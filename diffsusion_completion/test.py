import torch
import os
from PIL import Image
import numpy as np

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))
    d1 = torch.sum(thresh < 1.25).float() / thresh.numel()
    d2 = torch.sum(thresh < 1.25 ** 2).float() / thresh.numel()
    d3 = torch.sum(thresh < 1.25 ** 3).float() / thresh.numel()

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target) + 1e-9

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10': log10.item(), 'silog': silog.item()}

def cropping_img(pred, gt_depth):
    min_depth_eval = 0.1
    max_depth_eval = 80.0
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

def load_image(file_path):
    image = Image.open(file_path)
    return torch.tensor(np.array(image).astype(np.float32)).float()

def evaluate_depth_metrics(gt_folder, pred_folder):
    gt_files = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f))])
    pred_files = sorted([f for f in os.listdir(pred_folder) if os.path.isfile(os.path.join(pred_folder, f))])

    gt_files_dict = {os.path.splitext(f)[0]: f for f in gt_files}
    pred_files_dict = {os.path.splitext(f)[0]: f for f in pred_files}

    common_keys = set(gt_files_dict.keys()).intersection(set(pred_files_dict.keys()))

    metrics = {
        'd1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'log10': 0, 'silog': 0
    }
    num_files = len(common_keys)

    for key in common_keys:
        gt_file = gt_files_dict[key]
        pred_file = pred_files_dict[key]

        gt_depth = load_image(os.path.join(gt_folder, gt_file)) / 255.0
        pred_depth = load_image(os.path.join(pred_folder, pred_file)) / 255.0 + 1e-9
        print(pred_depth.shape, gt_depth.shape)
        pred_depth, gt_depth = cropping_img(pred_depth, gt_depth)
        print(pred_depth.shape, gt_depth.shape)
        result = eval_depth(pred_depth, gt_depth)

        for key in metrics:
            metrics[key] += result[key]

    for key in metrics:
        metrics[key] /= num_files

    return metrics

metrics = evaluate_depth_metrics('/home/disk/diffsusion_completion/train_gt', '/home/disk/diffsusion_completion/completed_gt')
print(metrics)