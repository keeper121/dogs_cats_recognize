import tensorflow as tf
import os
import glob
from model import model
import tensorflow.contrib.slim as slim
import random

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('iter_number', 2000, 'Max iteration number.')
flags.DEFINE_integer('epoch_count', 10, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('print_step', 1000, 'Print and validation step')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('test_dir', False, 'If true, uses fake data for unit testing.')
FLAGS = flags.FLAGS

image_shape = 224

def input_parser(img_path):
    # read the img from file
    # label = os.path.basename(img_path).split('.')[0]
    #label, _ = tf.decode_csv(img_path, [[""], [""]], '.')

    #tf

    img_file = tf.div(tf.cast(tf.image.decode_image(tf.read_file(img_path)), tf.float32), 255.0)
    img_file = tf.image.resize_bicubic([img_file], (image_shape, image_shape))[0] #  ????
    
    #   label = 1.0 if label == 'cat' else 0.0
    #label = 0.0
    label = tf.cond(tf.strings.regex_full_match(img_path, '.*cat.*'), lambda: 1.0, lambda: 0.0)

    return img_file, [label]

def train():
    tf.reset_default_graph()

    print glob.glob(FLAGS.train_dir + '*.*')
    #exit
    names = glob.glob(FLAGS.train_dir + '*.*')
    random.shuffle(names)
   
    tr_data = tf.data.Dataset.from_tensor_slices(names)
    tr_data = tr_data.map(input_parser)
    tr_data = tr_data.shuffle(buffer_size=100 * FLAGS.batch_size)
    tr_data = tr_data.batch(FLAGS.batch_size)

    shape = (tf.TensorShape([FLAGS.batch_size, image_shape, image_shape, 3]), tf.TensorShape([FLAGS.batch_size, 1]))
    types = (tf.float32, tf.float32)
    #types = (tf.float32)

    iterator_tr = tf.data.Iterator.from_structure(types, shape)

    images, labels = iterator_tr.get_next()

    predictions = model(images)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions))

    minimize_op = slim.learning.create_train_op(loss, 
                                                   tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate))

    training_init_op = iterator_tr.make_initializer(tr_data)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(training_init_op)
        sess.run(init)
        try:
            for i in xrange(FLAGS.iter_number):

                if i % FLAGS.print_step == 0:
                    _, comp_loss = sess.run([minimize_op, loss])
                    print 'iteration:', i
                    print 'loss', comp_loss
                    #print 'labels', pr_labels
                else:
                    sess.run(minimize_op)
            
        except tf.errors.OutOfRangeError as er:
            print er.message
            pass
    
    
    
if __name__ == '__main__':
    train()
