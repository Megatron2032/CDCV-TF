import tensorflow as tf
slim = tf.contrib.slim

class InceptionV3_facenet_112():
    def __init__(self,batch_seize=32,num_channels=3,image_size_h=112,image_size_w=112,trainable=True,resueable=False,BN_decay=0.99,BN_epsilon=0.00001,anchor_num=15,name="train/"):
        self.roi_data=[]
        self.s=tf.placeholder(tf.float32, shape=(batch_size, num_channels,image_size_h, image_size_w), name='tf_image')

        data = slim.conv2d_in_plane(inputs=self.s, num_outputs=8, kernel_size=[3, 3], stride=2,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'conv1_3x3_s1')

        data = slim.conv2d_in_plane(inputs=data, num_outputs=8, kernel_size=[3, 3], stride=1,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'conv1_3x3_s2')

        data = slim.conv2d_in_plane(inputs=data, num_outputs=16, kernel_size=[3, 3], stride=1,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'conv1_3x3_s3')

        data = slim.max_pool2d(inputs=data, kernel_size=[3, 3], stride=2,padding='SAME',data_format="NCHW",scope='pool1_3x3')

        data = slim.conv2d_in_plane(inputs=data, num_outputs=16, kernel_size=[1, 1], stride=1,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'conv4_3x3_reduce')

        data = slim.conv2d_in_plane(inputs=data, num_outputs=48, kernel_size=[3, 3], stride=1,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'conv4_3x3')

        data = slim.max_pool2d(inputs=data, kernel_size=[3, 3], stride=2,padding='SAME',data_format="NCHW",scope='pool2_3x3')

        data = self.small_inceptionv3(data,trainable=trainable,resueable=resueable,BN_decay=BN_decay,BN_epsilon=BN_epsilon,name=name+"inception1/")
        data = self.small_inceptionv3(data,trainable=trainable,resueable=resueable,BN_decay=BN_decay,BN_epsilon=BN_epsilon,name=name+"inception2/")

        data = slim.conv2d_transpose(inputs=data, num_outputs=32, kernel_size=[2, 2], stride=2,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope=name+'deconv')
        self.roi_data.append(data)

        data = slim.conv2d_in_plane(inputs=data, num_outputs=64, kernel_size=[3, 3], stride=1,padding='SAME',
                                    activation_fn=tf.nn.relu,data_format="NCHW",
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope='rpn_conv')

        rpn_cls_score = slim.conv2d_in_plane(inputs=data, num_outputs=anchor_num*2, kernel_size=[1, 1], stride=1,padding='SAME',
                                    data_format="NCHW",
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope='rpn_cls_score')

        rpn_bbox_pred = slim.conv2d_in_plane(inputs=data, num_outputs=anchor_num*4, kernel_size=[1, 1], stride=1,padding='SAME',
                                    data_format="NCHW",
                                    weights_initializer=initializers.xavier_initializer(),
                                    reuse=resueable,
                                    trainable=trainable,
                                    scope='rpn_bbox_pred')

        rpn_cls_score_reshape = tf.reshape(rpn_cls_score,[tf.shape(rpn_cls_score)[0],2,-1,tf.shape(rpn_cls_score)[3]],name='rpn_cls_score_reshape')
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape,axis=1,name='rpn_cls_prob')
        rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob,[tf.shape(rpn_cls_prob)[0],anchor_num*2,-1,tf.shape(rpn_cls_prob)[3]],name='rpn_cls_prob_reshape')

        self.rpn_cls_score_reshape=rpn_cls_score_reshape
        self.rpn_bbox_pred=rpn_bbox_pred
        self.rpn_cls_prob_reshape=rpn_cls_prob_reshape

        def small_inceptionv3(self,data,trainable=True,resueable=False,BN_decay=0.99,BN_epsilon=0.00001,name="inception/"):
            data_a1 = slim.conv2d_in_plane(inputs=data, num_outputs=16, kernel_size=[1, 1], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a1_1x1')

            data_a2 = slim.conv2d_in_plane(inputs=data, num_outputs=12, kernel_size=[1, 1], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a2_1x1')
            data_a2 = slim.conv2d_in_plane(inputs=data_a2, num_outputs=16, kernel_size=[5, 5], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a2_5x5')

            data_a3 = slim.conv2d_in_plane(inputs=data, num_outputs=16, kernel_size=[1, 1], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a3_1x1')
            data_a3 = slim.conv2d_in_plane(inputs=data_a3, num_outputs=24, kernel_size=[3, 3], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a3_3x3')
            data_a3 = slim.conv2d_in_plane(inputs=data_a3, num_outputs=24, kernel_size=[3, 3], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a3_3x3_2')

            data_a4 = slim.avg_pool2d(inputs=data4, kernel_size=[3, 3], stride=1,padding='SAME',data_format="NCHW",scope='inception_a4_pool')
            data_a4 = slim.conv2d_in_plane(inputs=data_a4, num_outputs=8, kernel_size=[1, 1], stride=1,padding='SAME',
                                        activation_fn=tf.nn.relu,data_format="NCHW",
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': trainable, 'epsilon':BN_epsilon, 'decay':BN_decay,'scale': True,'reuse':resueable,'trainable':trainable, 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                                        weights_initializer=initializers.xavier_initializer(),
                                        reuse=resueable,
                                        trainable=trainable,
                                        scope=name+'inception_a4_1x1')

            data=tf.concat(1, [data_a1,data_a2,data_a3,data_a4])
            return data
