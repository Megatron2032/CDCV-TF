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
sys.path.append('Net/')
sys.path.append('Layer/')
sys.path.append('Net/Layer/')
from network import *
from hyperparameter import *
from loss import *
import datetime

class framework():
	def __init__(self,env,optimizeMethod='adam'):
		self.optimizeMethod = optimizeMethod


		# Graph Related
		self.graph = tf.Graph()
		# 初始化
		self.s,self.pi,self.vf,self.logstd, \
		self.sample_action,self.ac_prob, \
		self.all_weights,self.pg_loss,self.vf_loss,self.approxkl,self.clipfrac, \
		self._train,self.pi_train,self.vf_train=self.define_graph()

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(graph=self.graph,config=config)
		if TRAIN:
			self.writer = tf.summary.FileWriter('Graphview/', self.graph)
		# 放在定义Graph之后，保存这张计算图
		self.saver = tf.train.Saver(self.all_weights)


	def define_graph(self):
		'''
		定义我的的计算图谱
		'''
		with self.graph.as_default():
			# Training computation.
			self.A = tf.placeholder(tf.float32, [None,self.ac_space], 'action')
			self.ADV = tf.placeholder(tf.float32, [None])
			self.R = tf.placeholder(tf.float32, [None])
			self.OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
			self.OLDVPRED = tf.placeholder(tf.float32, [None])
			self.LR = tf.placeholder(tf.float32, [])
			self.CLIPRANGE = tf.placeholder(tf.float32, [])
			self.Logstd=tf.placeholder(tf.float32, [1, self.ac_space])
			s,pi,vf,logstd,weights,summaries,pi_params,vf_params=self.DPPO.model(net_name='DPPO', reuse=False)

			if not use_train_logstd:
				logstd=logstd*0+self.Logstd

			sample_action = pi + tf.exp(logstd) * tf.random_normal(tf.shape(pi))
			ac_prob=0.5 * tf.reduce_sum(tf.square((sample_action - pi) / tf.exp(logstd)), axis=-1)+ 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(sample_action)[-1])+ tf.reduce_sum(logstd, axis=-1)
			pg_loss,vf_loss,approxkl,clipfrac,_train,pi_train,vf_train=self.loss(pi,vf,logstd,weights,pi_params,vf_params,Regularization='NULL')

			self.merged_train_weights_summary= tf.summary.merge(summaries)
			self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

			return s,pi,vf,logstd,sample_action,ac_prob,weights,pg_loss,vf_loss,approxkl,clipfrac,_train,pi_train,vf_train

	def loss(self,pi,vf,logstd,weights,pi_params,vf_params,Regularization='NULL'):
		pg_loss,vf_loss,approxkl,clipfrac,_train,pi_train,vf_train=loss_fun(self,pi,vf,logstd,weights,pi_params,vf_params,Regularization='NULL')
		return pg_loss,vf_loss,approxkl,clipfrac,_train,pi_train,vf_train

	
