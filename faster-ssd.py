#encoding: utf-8
from __future__ import print_function, division

# 第三方
import os
import sys
import pyglet
import time
import tensorflow as tf
import random
import numpy as np
from collections import deque
from config import Param
from hyperparameter import *
from loss import *
from Specific_net.inceptionv3_facenet import *
from layers.rpn_anchor_target import *
import datetime

class framework():
	def __init__(self,env,optimizeMethod='adam'):
		self.optimizeMethod = optimizeMethod

		self.Work_Param=Param("ZWW_face_detection_config_112inception_out8.json")

		self.train_batch=32
		self.val_batch=32
		self.test_batch=1
		self.input_image_size=[3,self.Work_Param.max_size,self.Work_Param.max_size]

		self.rpn_variable=[]
		self.rcnn_variable=[]
		# Graph Related
		self.graph = tf.Graph()
		# 初始化
		self.train=self.faster_ssd()

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(graph=self.graph,config=config)
		'''
		if TRAIN:
			self.writer = tf.summary.FileWriter('Graphview/', self.graph)
		# 放在定义Graph之后，保存这张计算图
		self.saver = tf.train.Saver(self.all_weights)
		'''

	def faster_ssd(self):
		'''
		定义faster-ssd的计算图谱
		'''
		with self.graph.as_default():
			# Training computation.
			self.small_rpn_train=InceptionV3_facenet_112(batch_seize=train_batch,num_channels=self.input_image_size[0],image_size_h=self.input_image_size[1],image_size_w=self.input_image_size[2],
													trainable=True,resueable=False,BN_decay=0.99,BN_epsilon=0.00001,anchor_num=15,name="train_rpn/")
			self.rpn_target_layer=anchor_target_layer(self.small_rpn_train.roi_data,batch=train_batch,img_height=self.input_image_size[1],img_width=self.input_image_size[2],FrcnnParam=self.Work_Param)


			'''
			self.small_rpn_val=InceptionV3_facenet_112(batch_seize=val_batch,num_channels=self.input_image_size[0],image_size_h=self.input_image_size[1],image_size_w=self.input_image_size[2],
													trainable=False,resueable=True,BN_decay=0.99,BN_epsilon=0.00001,anchor_num=15,name="val/")
			self.small_rpn_test=InceptionV3_facenet_112(batch_seize=test_batch,num_channels=self.input_image_size[0],image_size_h=self.input_image_size[1],image_size_w=self.input_image_size[2],
													trainable=False,resueable=True,BN_decay=0.99,BN_epsilon=0.00001,anchor_num=15,name="test/")
			'''
			#self.merged_train_weights_summary= tf.summary.merge(summaries)
			#self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
			for var in tf.trainable_variables():
				if var.op.name.find('train_rpn') > 0:
					self.rpn_variable.append(var)
			train=loss(self.small_rpn_train.rpn_cls_score_reshape,self.small_rpn_train.rpn_bbox_pred,self.rpn_variable)
			return train

	def loss(self,rpn_cls,rpn_reg,rpn_params,Regularization='NULL'):
		train=loss_fun(self,rpn_cls,rpn_reg,rpn_params,Regularization='NULL')
		return train
