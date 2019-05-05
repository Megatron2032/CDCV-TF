#encoding: utf-8
from __future__ import print_function, division

# 第三方
import os
import sys
import pyglet
import tensorflow as tf
import gym,roboschool
import random
import numpy as np
sys.path.append('env/')
from monitor import *
from subproc_vec_env import SubprocVecEnv
from vec_normalize import VecNormalize,VecNormalizeTest
from collections import deque
from hyperparameter import *
from agent import *
from hyperparameter import *


def run(agent,env):
	with agent.sess as session:
		tf.global_variables_initializer().run()

		#Load network
		if LOAD_NETWORK:
		   agent.load_network()

		### 训练
		if TRAIN:  # Train mode
			print('Start Training')
			for i in range(NUM_EPISODES): 
				agent.agent_run(i)
		else:  # Test mode
			from OpenGL import GLU
			print('Start Test')
			r=0
			env=gym.make(ENV_NAME)
			if ENV_NORMALIZE:
				running_mean = np.load('{}/mean.npy'.format(SAVE_NETWORK_PATH))
				running_var = np.load('{}/var.npy'.format(SAVE_NETWORK_PATH))
				env = VecNormalizeTest(env, running_mean, running_var)
			#env.render()
			env.reset()
			for _ in range(NUM_EPISODES_AT_TEST*EP_LEN):
				actions= agent.sess.run(agent.pi, {agent.s:agent.obs})
				agent.obs[:], rewards, agent.dones, infos = env.step(actions[0])
				env.render()
				r+=rewards
			print("all_reward=",r/NUM_EPISODES_AT_TEST)

if __name__ == '__main__':
	if not TRAIN:
		NWORK=1
		EARLY_RESET=False
	env = SubprocVecEnv([make_env(ENV_NAME,i,log_monitor=LOG_MONITOR) for i in range(NWORK)])
	
	if ENV_NORMALIZE:
		env = VecNormalize(env)
        	
	Agent = agent(env,optimizeMethod='adam')
	run(Agent,env)
	
#运行完后终端输入： tensorboard --logdir Graphview
