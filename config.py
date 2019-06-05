import random
import numpy as np
import json
from __init__ import *


class Param():
    def __init__(self,config_name):
        f = open(getDataPath()+"/config/"+config_name, encoding='utf-8')
        self.json_dic=json.load(f,strict=False)

        #data
        self.use_flipped=int(self.json_dic["use_flipped"])
        self.pre_ms_train=int(self.json_dic["pre-ms-train"])
        self.post_ms_train=int(self.json_dic["post-ms-train"])
        self.pixel_means=list(map(float,self.json_dic["pixel_means"].strip().split(",")))
        self.FP=int(self.json_dic["FP"])
        self.JQ=int(self.json_dic["JQ"])
        self.n_classes=int(self.json_dic["n_classes"])
        self.GT_num=int(self.json_dic["GT_num"])

        #train
        self.max_size=int(self.json_dic["max_size"])

        #rcnn
        self.batch_size=int(self.json_dic["batch_size"])
        self.min_scale=int(self.json_dic["min_scale"])
        self.fg_fraction=float(self.json_dic["fg_fraction"])
        self.fg_thresh=float(self.json_dic["fg_thresh"])
        self.bg_thresh_hi=float(self.json_dic["bg_thresh_hi"])
        self.bg_thresh_lo=float(self.json_dic["bg_thresh_lo"])
        self.bbox_reg=int(self.json_dic["bbox_reg"])
        self.bbox_thresh=float(self.json_dic["bbox_thresh"])
        self.bbox_normalize_targets=int(self.json_dic["bbox_normalize_targets"])
        self.bbox_inside_weights=list(map(float,self.json_dic["bbox_inside_weights"].strip().split(",")))
        self.bbox_normalize_means=list(map(float,self.json_dic["bbox_normalize_means"].strip().split(",")))
        self.bbox_normalize_stds=list(map(float,self.json_dic["bbox_normalize_stds"].strip().split(",")))

        #rpn
        self.rpn_positive_overlap=float(self.json_dic["rpn_positive_overlap"])
        self.rpn_negative_overlap=float(self.json_dic["rpn_negative_overlap"])
        self.use_self_overlap=int(self.json_dic["use_self_overlap"])
        self.rpn_clobber_positives=int(self.json_dic["rpn_clobber_positives"])
        self.rpn_fg_fraction=float(self.json_dic["rpn_fg_fraction"])
        self.rpn_batchsize=int(self.json_dic["rpn_batchsize"])
        self.rpn_nms_thresh=float(self.json_dic["rpn_nms_thresh"])
        self.rpn_pre_nms_top_n=int(self.json_dic["rpn_pre_nms_top_n"])
        self.rpn_post_nms_top_n=int(self.json_dic["rpn_post_nms_top_n"])
        self.rpn_min_size=int(self.json_dic["rpn_min_size"])
        self.rpn_bbox_inside_weights=list(map(float,self.json_dic["rpn_bbox_inside_weights"].strip().split(",")))
        self.rpn_positive_weight=float(self.json_dic["rpn_positive_weight"])
        self.rpn_allowed_border=int(self.json_dic["rpn_allowed_border"])


        #test
        self.test_max_size=int(self.json_dic["test_max_size"])
        #test rcnn
        self.test_score_thresh=float(self.json_dic["test_score_thresh"])
        self.test_nms=float(self.json_dic["test_nms"])
        self.test_bbox_reg=int(self.json_dic["test_bbox_reg"])
        #test rpn
        self.test_rpn_nms_thresh=float(self.json_dic["test_rpn_nms_thresh"])
        self.test_rpn_pre_nms_top_n=int(self.json_dic["test_rpn_pre_nms_top_n"])
        self.test_rpn_post_nms_top_n=int(self.json_dic["test_rpn_post_nms_top_n"])
        self.test_rpn_min_size=int(self.json_dic["test_rpn_min_size"])

        #anchor
        self.feat_stride=list(map(float,self.json_dic["feat_stride"].strip().split(",")))
        self.anchors=list(map(float,self.json_dic["anchors"].strip().replace('\n','').split(",")))
        self.config_n_anchors=list(map(int,self.json_dic["config_n_anchors"].strip().split(",")))
        self.max_gt_an_num=int(self.json_dic["max_gt_an_num"])

        self.eps=float(self.json_dic["eps"])
        self.inf=float(self.json_dic["inf"])




    def print_Param(self):
        print(self.json_dic)

if __name__ == '__main__':
    A=Param("ZWW_face_detection_config_112inception_out8.json")
    A.print_Param()
