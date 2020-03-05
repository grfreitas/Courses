import scipy.misc
import tensorflow as tf
import numpy as np
import time
import sys

class Config:
    data_path = "figures"
    model_path_train = ""
    model_path_test = "figures/checkpoint/model_2.ckpt" 
    output_path = "results"

    img_size = 424
    adjust_size = 424
    train_size = 424
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    save_per_epoch=1





import numpy as np
import tensorflow as tf
import scipy.misc

def batch_norm(x, scope):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=scope)

def conv2d(input, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, d_h, d_w, 1], padding='SAME'), bias)
        return conv

def deconv2d(input, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, weight, output_shape=output_shape, strides=[1, d_h, d_w, 1]), bias)
        return deconv

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def linear(input, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input, weight) + bias

def imread(path):
    return scipy.misc.imread(path)

def imsave(image, path):
    return scipy.misc.imsave(path, image)





import os

def load(path):
    for i in os.listdir(path):
        all = imread(path + "/" + i)
        img, cond = all[:,:conf.img_size], all[:,conf.img_size:]
        yield (img, cond, i)

def load_data():
    data = dict()
    data["train"] = lambda: load(conf.data_path + "/train")
    data["test"] = lambda: load(conf.data_path + "/test")
    return data




class CGAN(object):

    def __init__(self):
        self.image = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, conf.img_channel))
        self.cond = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, conf.img_channel))

        self.gen_img = self.generator(self.cond)

        pos = self.discriminator(self.image, self.cond, False)
        neg = self.discriminator(self.gen_img, self.cond, True)
        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
        neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))

        self.d_loss = pos_loss + neg_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg))) + \
                      conf.L1_lambda * tf.reduce_mean(tf.abs(self.image - self.gen_img))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

    def discriminator(self, img, cond, reuse):
        dim = len(img.get_shape())
        with tf.variable_scope("disc", reuse=reuse):
            image = tf.concat([img, cond], dim - 1)
            feature = conf.conv_channel_base
            h0 = lrelu(conv2d(image, feature, name="h0"))
            h1 = lrelu(batch_norm(conv2d(h0, feature*2, name="h1"), "h1"))
            h2 = lrelu(batch_norm(conv2d(h1, feature*4, name="h2"), "h2"))
            h3 = lrelu(batch_norm(conv2d(h2, feature*8, name="h3"), "h3"))
            h4 = linear(tf.reshape(h3, [1,-1]), 1, "linear")
        return h4

    def generator(self, cond):
        with tf.variable_scope("gen"):
            feature = conf.conv_channel_base
            e1 = conv2d(cond, feature, name="e1")
            e2 = batch_norm(conv2d(lrelu(e1), feature*2, name="e2"), "e2")
            e3 = batch_norm(conv2d(lrelu(e2), feature*4, name="e3"), "e3")
            e4 = batch_norm(conv2d(lrelu(e3), feature*8, name="e4"), "e4")
            e5 = batch_norm(conv2d(lrelu(e4), feature*8, name="e5"), "e5")
            e6 = batch_norm(conv2d(lrelu(e5), feature*8, name="e6"), "e6")
            e7 = batch_norm(conv2d(lrelu(e6), feature*8, name="e7"), "e7")
            e8 = batch_norm(conv2d(lrelu(e7), feature*8, name="e8"), "e8")

            size = conf.img_size
            num = [0] * 9
            for i in range(1,9):
                num[9-i]=size
                size =(size+1)/2

            d1 = deconv2d(tf.nn.relu(e8), [1,num[1],num[1],feature*8], name="d1")
            d1 = tf.concat([tf.nn.dropout(batch_norm(d1, "d1"), 0.5), e7], 3)
            d2 = deconv2d(tf.nn.relu(d1), [1,num[2],num[2],feature*8], name="d2")
            d2 = tf.concat([tf.nn.dropout(batch_norm(d2, "d2"), 0.5), e6], 3)
            d3 = deconv2d(tf.nn.relu(d2), [1,num[3],num[3],feature*8], name="d3")
            d3 = tf.concat([tf.nn.dropout(batch_norm(d3, "d3"), 0.5), e5], 3) 
            d4 = deconv2d(tf.nn.relu(d3), [1,num[4],num[4],feature*8], name="d4")
            d4 = tf.concat([batch_norm(d4, "d4"), e4], 3)
            d5 = deconv2d(tf.nn.relu(d4), [1,num[5],num[5],feature*4], name="d5")
            d5 = tf.concat([batch_norm(d5, "d5"), e3], 3) 
            d6 = deconv2d(tf.nn.relu(d5), [1,num[6],num[6],feature*2], name="d6")
            d6 = tf.concat([batch_norm(d6, "d6"), e2], 3)
            d7 = deconv2d(tf.nn.relu(d6), [1,num[7],num[7],feature], name="d7")
            d7 = tf.concat([batch_norm(d7, "d7"), e1], 3) 
            d8 = deconv2d(tf.nn.relu(d7), [1,num[8],num[8],conf.img_channel], name="d8")

            return tf.nn.tanh(d8)
            
            




def prepocess_train(img, cond,):
    img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    #img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    #cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    return img,cond

def prepocess_test(img, cond):

    img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.img_size , conf.img_size , conf.img_channel)
    cond = cond.reshape(1, conf.img_size , conf.img_size , conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

def train():
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    start_time = time.time()
    if not os.path.exists(conf.data_path + "/checkpoint"):
        os.makedirs(conf.data_path + "/checkpoint")
    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if conf.model_path_train == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path_train)
        for epoch in xrange(conf.max_epoch):
            counter = 0
            train_data = data["train"]()
            for img, cond, name in train_data:
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:img, model.cond:cond})
                counter += 1
                if counter % 50 ==0:
                    print ("Epoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f") \
                      % (epoch, counter, time.time() - start_time, m, M)
            if (epoch + 1) % conf.save_per_epoch == 0:
                save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))
                print ("Model saved in file: %s") % save_path
                test_data = data["test"]()
                for img, cond, name in test_data:
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])
                    gen_img = (gen_img + 1.) * 127.5
                    image = np.concatenate((gen_img, cond), axis=1).astype(np.int)
                    imsave(image, conf.output_path + "/%s" % name)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    train()
