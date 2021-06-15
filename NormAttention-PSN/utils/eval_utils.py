import torch
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid   = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean  = ang_valid.sum() / valid
    n_err_med  = ang_valid.median()  #clean
    n_acc_11   = (ang_valid < 11.25).sum().float() / valid.float()
    n_acc_30   = (ang_valid < 30).sum().float() / valid.float()
    n_acc_45   = (ang_valid < 45).sum().float() / valid.float()

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), 'n_err_med': n_err_med.item(),
            'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()} #clean
    angular_error_map = {'angular_map': angular_map} #clean
    return value, angular_map

def colorMap(diff, thres=90): #clean
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = torch.from_numpy(mpl.cm.jet(diff_norm.numpy()))[:,:,:, :3]
    return diff_cm.permute(0,3,1,2).clone().float()