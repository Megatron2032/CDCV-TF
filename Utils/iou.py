import tensorflow as tf
import numpy as np

#box1:anchors  box2:gts
def compute_iou(box1,box2):
    in_h= tf.math.minimum(tf.expand_dims(box1[:,2],0), tf.expand_dims(box2[:,2],-1)) - tf.math.maximum(tf.expand_dims(box1[:,0],0), tf.expand_dims(box2[:,0],-1))
    in_w = tf.math.minimum(tf.expand_dims(box1[:,3],0), tf.expand_dims(box2[:,3],-1)) - tf.math.maximum(tf.expand_dims(box1[:,1],0), tf.expand_dims(box2[:,1],-1))
    inter = tf.math.maximum(0.,in_h)*tf.math.maximum(0.,in_w)
    union = tf.expand_dims((box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]),0) + tf.expand_dims((box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]),-1)-inter
    iou = inter / union
    return iou


if __name__ == '__main__':
    box1=[[0,0,1,1],[1,1,2,2],[2,2,3,3]]
    box2=[[0,0,4,4],[0.5,0.5,1.5,1.5],[-10,-10,-5,-5]]


    graph = tf.Graph()
    with graph.as_default():
        iou=compute_iou(tf.constant(value=box1,dtype=tf.float32),tf.constant(value=box2,dtype=tf.float32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(graph=graph,config=config)
    print(sess.run(iou))
