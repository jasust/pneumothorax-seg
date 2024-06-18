import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import slim


class Model:

    def __init__(self, name, size=(128, 128), training=False, test=False, gender='B'):

        self.DEBUG = False
        self.test = test
        self.training = training
        self.gender = gender

        self.height = size[0]
        self.width = size[1]
        self.order = []
        self.order0 = []
        self.order1 = []

        self.num_epochs = 35
        self.batch_size = 12
        self.display_step = 96
        self.point_step = 32

        self.dataset_x = []
        self.dataset_y = []
        self.validset_x = []
        self.validset_y = []
        self.testset_x = []

        self.loss_t = []
        self.acc_t = []
        self.loss_v = []
        self.acc_v = []

        self.folder_name = 'results' + os.sep + name + '_' + str(size[0]) + '_' + gender
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        print("Initialising model...", end=' ')

        def conv_net(data, is_training):

            with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=slim.batch_norm,
                                normalizer_params={'decay': 0.999, 'is_training': True, 'updates_collections': None,
                                                   'trainable': is_training},
                                weights_initializer=slim.initializers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                conv1 = slim.repeat(data, 2, slim.conv2d, 32, [3, 3], scope='conv1')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                conv2 = slim.repeat(pool1, 2, slim.conv2d, 64, [5, 5], scope='conv2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                conv3 = slim.repeat(pool2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                conv4 = slim.repeat(pool3, 2, slim.conv2d, 256, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

                # conv1 = slim.conv2d(data, 32, [3, 3], scope='conv1')
                # pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                # conv2 = slim.conv2d(pool1, 64, [5, 5], scope='conv2')
                # pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                # conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3')
                # pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                # conv4 = slim.conv2d(pool3, 256, [3, 3], scope='conv4')
                # pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

                fc1 = slim.fully_connected(tf.contrib.layers.flatten(pool4), 256, scope='fc1')
                # dp1 = slim.dropout(fc1, 0.5, scope='dropout1')
                fc2 = slim.fully_connected(fc1, 64, scope='fc2')
                output = slim.fully_connected(fc2, 2, activation_fn=tf.nn.sigmoid, scope='output')
                # TODO: add dropout if needed

            return output

        self.x = tf.placeholder("float", [None, self.height, self.width, 1])
        self.y = tf.placeholder("int32", [None])

        logits = conv_net(self.x, is_training=self.training)

        self.pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        self.sm = tf.nn.softmax(logits)
        self.acc = tf.count_nonzero(tf.equal(self.pred, self.y))
        self.acc = self.acc / self.batch_size

        # class_weight = tf.constant([0.25, 0.75])
        # weighted_logits = tf.multiply(logits, class_weight)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)

        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, use_locking=False).minimize(self.loss)

        print("Done.")

    def load(self):

        print('Loading data...', end=' ')

        if self.training:
            train_input = np.load('data' + os.sep + 'train_input_' + str(self.height) + '.npy')
            train_input_aug = np.load('data' + os.sep + 'train_input_augmented_' + str(self.height) + '.npy')
            train_output = np.load('data' + os.sep + 'train_class_' + str(self.height) + '.npy')
            train_output_aug = np.load('data' + os.sep + 'train_class_' + str(self.height) + '.npy')
            train_gender = np.load('data' + os.sep + 'train_gender_' + str(self.height) + '.npy')
            train_class = np.load('data' + os.sep + 'train_class_' + str(self.height) + '.npy')
            train_classes = np.hstack((train_class, train_class))

            # train_input_aug = train_input_aug[train_class == 1]
            # train_output_aug = train_output_aug[train_class == 1]

            # TODO: gledaj sta augmentujes ==1 je upitno svuda i train_class kao u u-net
            if self.gender == 'M':
                train_input = train_input[train_gender[train_class == 1]]
                train_input_aug = train_input_aug[train_gender[train_class == 1]]
                train_output = train_output[train_gender[train_class == 1]]
                train_output_aug = train_output_aug[train_gender[train_class == 1]]
            elif self.gender == 'F':
                train_input = train_input[~train_gender[train_class == 1]]
                train_input_aug = train_input_aug[~train_gender[train_class == 1]]
                train_output = train_output[~train_gender[train_class == 1]]
                train_output_aug = train_output_aug[~train_gender[train_class == 1]]

            images = np.vstack((train_input, train_input_aug))
            classes = np.hstack((train_output, train_output_aug))

            images = images.astype(np.float32) * (1. / 255) - 0.5

            self.dataset_x = np.reshape(images, (-1, self.height, self.width, 1))
            self.dataset_y = classes
            self.order = list(range(0, len(self.dataset_y)))
            self.order1 = list(np.where(train_classes == 1)[0])
            self.order0 = list(np.where(train_classes == 0)[0])

        valid_input = np.load('data' + os.sep + 'valid_input_' + str(self.height) + '.npy')
        valid_output = np.load('data' + os.sep + 'valid_class_' + str(self.height) + '.npy')
        valid_gender = np.load('data' + os.sep + 'valid_gender_' + str(self.height) + '.npy')

        if self.gender == 'M':
            valid_input = valid_input[valid_gender]
            valid_output = valid_output[valid_gender]
        elif self.gender == 'F':
            valid_input = valid_input[~valid_gender]
            valid_output = valid_output[~valid_gender]

        valid_input = valid_input.astype(np.float32) * (1. / 255) - 0.5

        self.validset_x = np.reshape(valid_input, (-1, self.height, self.width, 1))
        self.validset_y = valid_output

        if self.test:
            test_input = np.load('data' + os.sep + 'test_input_' + str(self.height) + '.npy')
            test_gender = np.load('data' + os.sep + 'test_gender_' + str(self.height) + '.npy')

            if self.gender == 'M':
                test_input = test_input[test_gender]
            elif self.gender == 'F':
                test_input = test_input[~test_gender]

            test_input = test_input.astype(np.float32) * (1. / 255) - 0.5
            self.testset_x = np.reshape(test_input, (-1, self.height, self.width, 1))

        print('Done.')

    def train(self):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.num_epochs):

                step = 1
                self.order = self.order1 + random.sample(self.order0, len(self.order1))
                random.shuffle(self.order)
                start_time = time.time()

                while step * self.batch_size <= len(self.order):

                    batch_x = [self.dataset_x[k] for k in self.order[(step-1) * self.batch_size:step * self.batch_size]]
                    batch_y = [self.dataset_y[k] for k in self.order[(step-1) * self.batch_size:step * self.batch_size]]

                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                    if step % self.point_step == 0:

                        loss_value, acc_value = sess.run([self.loss, self.acc],
                                                         feed_dict={self.x: batch_x, self.y: batch_y})
                        self.loss_t.append(loss_value)
                        self.acc_t.append(100 * acc_value)

                        if step % self.display_step == 0:
                            print("Epoch %d, step %d: loss = %f, acc = %.4f" % (i+1, step, loss_value, acc_value))

                    step += 1

                duration = time.time() - start_time
                print("Duration of training for this epoch: %.2fs" % duration)

                acc_valid = 0
                loss_valid = 0
                for j in range(len(self.validset_x) // self.batch_size):
                    loss, acc = sess.run([self.loss, self.acc],
                                         feed_dict={self.x: self.validset_x[j*self.batch_size:(j+1)*self.batch_size],
                                                    self.y: self.validset_y[j*self.batch_size:(j+1)*self.batch_size]})
                    acc_valid += acc
                    loss_valid += loss

                acc_valid /= len(self.validset_x) // self.batch_size
                loss_valid /= len(self.validset_x) // self.batch_size

                acc_train = np.mean(self.acc_t[-(step//self.point_step)+1:])
                loss_train = np.mean(self.loss_t[-(step//self.point_step)+1:])
                self.acc_v.append((acc_train, acc_valid*100))
                self.loss_v.append((loss_train, loss_valid))

                print('Training accuracy:   %.2f, Training loss:   %.4f' % (self.acc_v[i][0], self.loss_v[i][0]))
                print('Validation accuracy: %.2f, Validation loss: %.4f' % (self.acc_v[i][1], self.loss_v[i][1]))

                if i and (np.min(self.loss_v, axis=0)[1] >= self.loss_v[i][1] or
                          np.max(self.acc_v, axis=0)[1] <= self.acc_v[i][1]):
                    saver.save(sess, self.folder_name + os.sep + 'model.ckpt', i+1)

            print("Optimization Finished...")

        plt.plot(self.acc_t)
        plt.title("Accuracy in %")
        plt.xlabel('Iteration')
        plt.savefig(self.folder_name + os.sep + 'acc_t.png')
        plt.close()

        plt.plot(self.loss_t)
        plt.title("Loss function")
        plt.xlabel('Iteration')
        plt.savefig(self.folder_name + os.sep + 'loss_t.png')
        plt.close()

        plt.plot([self.acc_v[k][0] for k in range(self.num_epochs)])
        plt.plot([self.acc_v[k][1] for k in range(self.num_epochs)])
        plt.xlabel('Epoch')
        plt.legend(('training accuracy', 'validation accuracy'))
        plt.title("Accuracy [%] of training and validation data")
        plt.savefig(self.folder_name + os.sep + 'acc_v.png')
        plt.close()

        plt.plot([self.loss_v[k][0] for k in range(self.num_epochs)])
        plt.plot([self.loss_v[k][1] for k in range(self.num_epochs)])
        plt.xlabel('Epoch')
        plt.legend(('training loss', 'validation loss'))
        plt.title("Loss function of training and validation data")
        plt.savefig(self.folder_name + os.sep + 'loss_v.png')
        plt.close()


if __name__ == '__main__':

    model = Model(name='class_v2', training=True)
    model.load()
    model.train()
