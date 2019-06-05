#encoding: utf-8
from __future__ import print_function, division

import os
import sys
import time
import tensorflow as tf
import random
import numpy as np
sys.path.append("..")
from Utils.iou import compute_iou
from Utils.util import *
from hyperparameter import *

class anchor_target_layer():
    def __init__(self,rpn_out_layers,batch,img_height,img_width,FrcnnParam):
        self.FrcnnParam=FrcnnParam
        self.GT=tf.placeholder(tf.float32,shape=[None,4],name='groundtruth')
        self.img_width=img_width
        self.img_height=img_height
        self.batch=batch
        self.width=[]
        self.height=[]
        self.rpn_out_layers_num=len(rpn_out_layers)
        for data in rpn_out_layers:
            self.width.append(data.shape.as_list()[1])
            self.height.append(data.shape.as_list()[2])
        self.config_n_anchors_=FrcnnParam.config_n_anchors;
        self.border_ = FrcnnParam.rpn_allowed_border
        self.max_num = FrcnnParam.max_gt_an_num
        self.bounds = [-self.border_, -self.border_, self.img_width + self.border_, self.img_height + self.border_]
        self.feat_stride_=FrcnnParam.feat_stride
        self.anchors_=FrcnnParam.anchors
        print(self.config_n_anchors_)
        print(self.anchors_)
        self.anchors=[]
        self.inds_inside=[]
        config_n_anchors_=0
        num_config_n_anchors=0
        self.global_num=0
        for layer in range(self.rpn_out_layers_num):
            height=self.height[layer]
            width=self.width[layer]
            num_config_n_anchors+=config_n_anchors_
            config_n_anchors_=self.config_n_anchors_[layer]
            for h in range(height):
                for w in range(width):
                    for k in range(config_n_anchors_):
                        x1 = w * self.feat_stride_[layer] + self.anchors_[(k+num_config_n_anchors) * 4 + 0]
                        y1 = h * self.feat_stride_[layer] + self.anchors_[(k+num_config_n_anchors) * 4 + 1]
                        x2 = w * self.feat_stride_[layer] + self.anchors_[(k+num_config_n_anchors) * 4 + 2]
                        y2 = h * self.feat_stride_[layer] + self.anchors_[(k+num_config_n_anchors) * 4 + 3]
                        self.global_num+=1
                        if (x1 >= self.bounds[0] and y1 >= self.bounds[1] and x2 < self.bounds[2] and y2 < self.bounds[3]):
                             self.inds_inside.append([layer,h,w,k,self.global_num-1])
                             self.anchors.append([x1, y1, x2, y2])

        self.label=-1 * np.ones((self.batch,1,self.global_num,1))
        self.box_target=np.zeros((self.batch,4,self.global_num,1))
        self.bbox_inside_weights=np.zeros((self.batch,4,self.global_num,1))
        self.bbox_outside_weights=np.zeros((self.batch,4,self.global_num,1))

        self.anchors_tf = tf.constant(value=self.anchors,dtype=tf.float32)

    def forward(self,session,GT): #GT:cls box diff landmark jq batch_index
        ious=compute_iou(self.anchors_tf,self.GT)
        self.ious=session.run(ious,feed_dict={self.GT:GT[:,1:5]})
        for batch_index in range(self.batch):
            gt=[]
            gt_index=[]
            labels=np.array([-1 for x in range(ious.shape[1])])
            max_overlaps=np.array([-1.0 for x in range(ious.shape[1])])
            argmax_overlaps=np.array([-1 for x in range(ious.shape[1])])
            for j in range(GT.shape[0]):
                if int(GT[j][-1])==batch_index:
                    gt.append(GT[j][:-1])
                    gt_index.append(j)
            gt_anchor_num=np.array([0 for x in range(len(gt))])
            gt_an=[[] for x in range(len(gt))]
            anchor_iou=self.ious[gt_index]
            argmax_overlaps=np.argmax(anchor_iou,axis=0)
            max_overlaps=np.max(anchor_iou,axis=0)

            #fg label: for each gt, sort anchors with overlap
            ##############policy2#################
            policy_bg_choice=[]
            policy_fg_choice=[]
            choice_anchors=[]
            for i in range(max_overlaps.shape[0]):
                if max_overlaps[i]>=0.33 and 0<=gt[argmax_overlaps[i]][5]<=1:  #0<=diff<=1
                    choice_anchors.append(i)
                if max_overlaps[i]<self.FrcnnParam.rpn_negative_overlap:
                    policy_bg_choice.append(i)

            new_choice_anchors=sorted(choice_anchors, key=lambda x:max_overlaps[x], reverse=True)

            new_choice_anchors_copy=list(new_choice_anchors)
            for i in new_choice_anchors:
                    if max_overlaps[i]>0.5 and labels[i]!=1 and gt_anchor_num[argmax_overlaps[i]]<self.max_num:
                        labels[i]=1
                        policy_fg_choice.append(i)
                        gt_anchor_num[argmax_overlaps[i]]+=1
                        new_choice_anchors_copy.remove(i)

            for i in new_choice_anchors_copy:
                gt_an[argmax_overlaps[i]].append(i)

            gt_index=0
            for gt_max in gt_an:
                while(gt_anchor_num[gt_index]<self.max_num and len(gt_max)>0):
                    i=random.choice(gt_max)
                    if labels[i] !=1:
                        gt_max.remove(i)
                        policy_fg_choice.append(i)
                        labels[i]=1
                        gt_anchor_num[gt_index]+=1
                gt_index +=1

            real_fg_choice=[]
            if not FocalLoss:
                num_fg = int(self.FrcnnParam.rpn_fg_fraction * self.FrcnnParam.rpn_batchsize)
                if num_fg>=len(policy_fg_choice):
                    real_fg_choice=list(policy_fg_choice)
                else:
                    real_fg_choice=random.sample(policy_fg_choice,num_fg)
                    for i in policy_fg_choice:
                        if i not in real_fg_choice:
                            labels[i]=-1
            else:
                real_fg_choice=list(policy_fg_choice)

            #bg label
            num_bg = self.FrcnnParam.rpn_batchsize-num_fg
            if not FocalLoss:
                if num_bg>=len(policy_bg_choice):
                    real_bg_choice=list(policy_bg_choice)
                else:
                    real_bg_choice=random.sample(policy_bg_choice,num_bg)
            else:
                real_bg_choice=list(policy_bg_choice)
            labels[np.array(real_bg_choice)]=0
            ######################################################
            #label
            #print(np.array(self.inds_inside)[:,-1])
            self.label[batch_index,0,np.array(self.inds_inside)[:,-1],0]=labels

            #box
            box1=np.array(self.anchors)[np.array(real_bg_choice)]
            box2=np.array(gt)[np.array(argmax_overlaps)[np.array(real_bg_choice)]]
            print(gt)
            box_target=transform_box(box1,box2[:,1:5]) #box1->box2
            self.box_target[batch_index,0,np.array(real_bg_choice),0]=box_target[0]
            self.box_target[batch_index,1,np.array(real_bg_choice),0]=box_target[1]
            self.box_target[batch_index,2,np.array(real_bg_choice),0]=box_target[2]
            self.box_target[batch_index,3,np.array(real_bg_choice),0]=box_target[3]

            self.bbox_inside_weights[batch_index,:,np.array(real_bg_choice),0]=1
            self.bbox_outside_weights[batch_index,:,np.array(real_bg_choice),0]=1

        return [self.label,self.box_target,self.bbox_inside_weights,self.bbox_outside_weights]


    def anchor_policy_S3FD(self,):
        pass


if __name__ == '__main__':
    import cv2
    from config import Param
    img=cv2.imread("../img/test.png")
    w=img.shape[0]
    h=img.shape[1]
    alpha=112.0/max(w,h)
    img=cv2.resize(img,(int(w*alpha),int(h*alpha)),interpolation=cv2.INTER_AREA)
    GT=np.array([[0,0,0,40,40,0,0],[0,0.5,0.5,10.5,10.5,0,0]])
    rpn_out_layers=[tf.constant(-1.0, shape=[1,img.shape[0]/4,img.shape[1]/4, 1])]
    image=tf.expand_dims(tf.constant(img),0)
    A=Param("ZWW_face_detection_config_112inception_out8.json")
    target_layer=anchor_target_layer(rpn_out_layers=rpn_out_layers,batch=1,img_height=h,img_width=w,FrcnnParam=A)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    top=target_layer.forward(sess,GT)
    print(top[2])
