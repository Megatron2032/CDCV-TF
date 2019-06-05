import numpy as np


def transform_box(box1,box2):
    ex_width = box1[:,2] - box1[:,0] + 1;
    ex_height = box1[:,3] - box1[:,1] + 1;
    ex_ctr_x = box1[:,0] + 0.5 * ex_width;
    ex_ctr_y = box1[:,1] + 0.5 * ex_height;
    gt_widths = box2[:,2] - box2[:,0] + 1;
    gt_heights = box2[:,3] - box2[:,1] + 1;
    gt_ctr_x = box2[:,0] + 0.5 * gt_widths;
    gt_ctr_y = box2[:,1] + 0.5 * gt_heights;
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
    targets_dw = np.log(gt_widths / ex_width);
    targets_dh = np.log(gt_heights / ex_height);
    return np.array([targets_dx,targets_dy,targets_dw,targets_dh])
