#encoding: utf-8
from __future__ import print_function, division

# 第三方
import os
import pyglet
import tensorflow as tf
import random
import numpy as np
from collections import deque
from hyperparameter import *


def loss_fun(self,pi,vf,logstd,all_weights,pi_params,vf_params,Regularization='NULL'):
	with tf.name_scope('loss'):
		neglogpac = 0.5 * tf.reduce_sum(tf.square((self.A - pi) / tf.exp(logstd)), axis=-1)+ 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.A)[-1])+ tf.reduce_sum(logstd, axis=-1)
		vfclipped = self.OLDVPRED + tf.clip_by_value(vf - self.OLDVPRED, - self.CLIPRANGE, self.CLIPRANGE)
	with tf.name_scope('vf_loss'):
		vf_losses1 = tf.square(vf - self.R)
		vf_losses2 = tf.square(vfclipped - self.R)
		vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
	with tf.name_scope('ratio'):
		ratio = tf.exp(self.OLDNEGLOGPAC - neglogpac)
	with tf.name_scope('pg_loss'):
		pg_losses = -self.ADV * ratio
		pg_losses2 = -self.ADV * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE, 1.0 + self.CLIPRANGE)
		pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
	with tf.name_scope('kl'):
		approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.OLDNEGLOGPAC))
	with tf.name_scope('clipfrac'):
		clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.CLIPRANGE)))
	with tf.name_scope('loss'):
		#loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
		loss = pg_loss + vf_loss * self.vf_coef

	if Regularization=='L2':
		# L2 regularization for the fully connected parameters
		regularization = 0.0
		for weights in all_weights:
			regularization += tf.nn.l2_loss(weights)
		regularization=self.weight_decay * regularization
		loss += regularization      #修正后的L2正则化,去除学习率与权重衰减率的耦合,增大学习空间(未修正)


	# Optimizer.
	with tf.name_scope('optimizer'):
		if(self.optimizeMethod=='gradient'):
			optimizer = tf.train.GradientDescentOptimizer(self.LR)
			optimizer2 = tf.train.GradientDescentOptimizer(VALUE_LR)
		elif(self.optimizeMethod=='momentum'):
			optimizer = tf.train.MomentumOptimizer(self.LR, MOMENTUM)
			optimizer2 = tf.train.MomentumOptimizer(VALUE_LR, MOMENTUM)
		elif(self.optimizeMethod=='adam'):
			optimizer = tf.train.AdamOptimizer(self.LR, epsilon=1e-5)
			optimizer2 = tf.train.AdamOptimizer(VALUE_LR, epsilon=1e-5)
		elif(self.optimizeMethod=='RMSProp'):
			optimizer = tf.train.RMSPropOptimizer(self.LR,momentum=MOMENTUM,epsilon=MIN_GRAD)
			optimizer2 = tf.train.RMSPropOptimizer(VALUE_LR,momentum=MOMENTUM,epsilon=MIN_GRAD)
		else:
			raise Exception("error:no optimizer match,optimizer=[gradient,momentum,adam,RMSProp]")
	#shared net
	with tf.variable_scope('model'):
		params = tf.trainable_variables()
	grads = tf.gradients(loss, params)
	if max_grad_norm is not None:
		grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
	grads = list(zip(grads, params))
	_train = optimizer.apply_gradients(grads)
	#independent net
	grads_pi = tf.gradients(pg_loss, pi_params)
	grads_vf = tf.gradients(vf_loss, vf_params)
	if max_grad_norm is not None:
		grads_pi, _grad_norm_pi = tf.clip_by_global_norm(grads_pi, max_grad_norm)
		grads_vf, _grad_norm_vf = tf.clip_by_global_norm(grads_vf, max_grad_norm)
	grads_pi = list(zip(grads_pi, pi_params))
	grads_vf = list(zip(grads_vf, vf_params))
	pi_train = optimizer.apply_gradients(grads_pi)
	vf_train = optimizer2.apply_gradients(grads_vf)

	return pg_loss,vf_loss,approxkl,clipfrac,_train,pi_train,vf_train
