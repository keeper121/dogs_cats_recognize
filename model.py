import tensorflow as tf
import tensorflow.contrib.slim as slim
from vgg19 import vgg19

vgg_model = vgg19('/home/keeper121/entropix/implementation_py/entropix/evaler/models/vgg19.npy',
                  need_fc=False)

def model(image):

    # TODO think about resize
    vgg_model.build(image * 255.0)

    dense = slim.flatten(vgg_model.pool5)
    fc1 = slim.fully_connected(dense,
                               1024,
                               activation_fn=lambda x: tf.nn.leaky_relu(x, 0.1))


    
    fc2 = slim.fully_connected(fc1,
                               1,
                               activation_fn=tf.nn.sigmoid)

    return fc2

    
