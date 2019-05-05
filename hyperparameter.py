ENV_NAME ='half_robot' #'RoboschoolHumanoidFlagrunHarder-v1'  # Environment name
NUM_EPISODES = 6000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
LOG_MONITOR=False
ENV_NORMALIZE=True
EARLY_RESET=True
NWORK=32
BATCH_SIZE = 4096
UPDATE_NUM=15
EP_LEN=512
RECURRENT_VERSION=None   #recurrent version or nonrecurrent version
#learning
SHARED_NET=True
use_train_logstd=True
LOGSTD_START=-0.7  #use_train_logstd=False
LOGSTD_END=-1.6  #use_train_logstd=False
VF_COEF=0.5     #if use SHARED_NET
LEARNING_RATE =0.0001  # Learning rate
VALUE_LR=0.001		 # value net Learning rate, SHARED_NET=False
DECAY_STEP=100     #multistep
MOMENTUM = 0.95  # Momentum
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the update
WEIGHT_DECAY=0.001  #L2
CLIPRANGE=0.2
KL=0.014
TD='lam'  #'0','mc','lam',n(1-EP_LEN)
INIT_TARG=0.01
max_grad_norm=0.5
USE_TARG_DECLINE=False
ADVS_MEAN=True
LR_MODE=0   #0:normal, 1:decay, 2:adaptive kl
CLIP_MODE=0   #0:Constant clip, 1:decay clip
GAMMA = 0.99  # Discount factor
LAM=0.95
#save
SAVE_EPISODES = 100  # The frequency with which the network is saved
LOAD_NETWORK = False
TRAIN = 1
SAVE_NETWORK_PATH = "saved_networks/" + ENV_NAME
NUM_EPISODES_AT_TEST = 3  # Number of episodes the agent plays at test time
