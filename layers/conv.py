#encoding: utf-8
from __future__ import print_function, division
import tensorflow as tf

def conv(self, data_flow,kernel_size,stride,in_depth,out_depth,pad,bias,activation,name,network_name):

	Name=network_name+'_'+name
	with tf.name_scope(Name):
		with tf.name_scope(Name + '_model'):
			weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, in_depth, out_depth], stddev=0.1), name=Name+'_weights')
			biases = tf.Variable(tf.constant(0., shape=[out_depth]), name=Name+'_biases')
			self.all_weights.append(weights)
			self.all_weights.append(biases)
			self.train_summaries.append(tf.summary.histogram(name+'_weights', weights))
			self.train_summaries.append(tf.summary.histogram(name+'_biases', biases))

		with tf.name_scope(Name+'convolution'):
			# default SAME padding
			data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, stride, stride, 1], padding=pad)
			if bias:
				data_flow=data_flow+biases
			if activation == 'relu':
				data_flow = tf.nn.relu(data_flow)
			elif activation =='tanh':
				data_flow =tf.nn.tanh(data_flow)
			elif activation =='sigmod':
				data_flow =tf.nn.sigmoid(data_flow)
			elif activation == 'None':
				print("{} / {} has no activation".format(network_name,name))
			else:
				raise Exception('Activation Func can only be Relu right now. You passed', 'activation')
	return data_flow,weights,biases
