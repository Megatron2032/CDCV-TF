#encoding: utf-8
from __future__ import print_function, division
import tensorflow as tf

def fc(self, data_flow,in_num_nodes, out_num_nodes, bias, activation, name, network_name):
	Name=network_name+'_'+name
	with tf.name_scope(Name):
		with tf.name_scope(Name+'_model'):
			weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
			biases = tf.Variable(tf.constant(0., shape=[out_num_nodes]))
			self.all_weights.append(weights)
			self.all_weights.append(biases)
			self.train_summaries.append(tf.summary.histogram(name+'_weights', weights))
			self.train_summaries.append(tf.summary.histogram(name+'_biases', biases))

		with tf.name_scope(Name+'fc'):
			data_flow = tf.matmul(data_flow, weights) 
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
