import os
import cv2
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt
from model import Model

DEBUG = False
im_height = 1024
im_width = 1024
size = (128, 128)

name = 'unet_v2'
version = 38

sess = tf.InteractiveSession()
model = Model(name=name, test=True)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(model.folder_name)
saver.restore(sess, ckpt.model_checkpoint_path)


def dice(y_true, y_pred):

    md = 0
    y_true_f = np.reshape(y_true, (y_true.shape[0], -1))
    y_pred_f = np.reshape(y_pred, (y_pred.shape[0], -1))
    for i in range(y_pred_f.shape[0]):
        union = np.sum(y_true_f[i]) + np.sum(y_pred_f[i])
        if union == 0:
            md += 1.0
        else:
            md += 2.*np.dot(y_true_f[i], y_pred_f[i])/union

    return md/y_pred_f.shape[0]

print('Loading data...', end=' ')
valid_input = np.load('data' + os.sep + 'valid_input_' + str(size[0]) + '.npy')
valid_output = np.load('data' + os.sep + 'valid_output_' + str(size[0]) + '.npy')
valid_gender = np.load('data' + os.sep + 'valid_gender_' + str(size[0]) + '.npy')

images = valid_input.astype(np.float32) * (1. / 255) - 0.5
validset_x = np.reshape(images, (-1, size[0], size[1], 1))
classes = (valid_output.astype(np.float32) * (1. / 255)).astype(np.int32)
validset_y = np.reshape(classes, (-1, size[0], size[1], 1))

test_input = np.load('data' + os.sep + 'test_input_' + str(size[0]) + '.npy')
test_output = np.load('data' + os.sep + 'test_output_' + str(size[0]) + '.npy')
test_gender = np.load('data' + os.sep + 'test_gender_' + str(size[0]) + '.npy')
test_id = np.load('data' + os.sep + 'test_id_' + str(size[0]) + '.npy')

test_input_og = np.load('data' + os.sep + 'test_input_' + str(1024) + '.npy')
test_output_og = np.load('data' + os.sep + 'test_output_' + str(1024) + '.npy')

test_input = test_input.astype(np.float32) * (1. / 255) - 0.5
testset_x = np.reshape(test_input, (-1, size[0], size[1], 1))
classes = (test_output.astype(np.float32) * (1. / 255)).astype(np.int32)
testset_y = np.reshape(classes, (-1, size[0], size[1], 1))

print('Done.')

pred_valid = np.zeros((len(validset_y), size[0], size[1], 1), dtype=np.int32)
for j in range(len(validset_y)//model.batch_size):
    pred_valid[j*model.batch_size:(j+1)*model.batch_size] = \
        sess.run(model.pred, feed_dict={model.x: validset_x[j*model.batch_size:(j+1)*model.batch_size]})

val_acc = dice(validset_y, pred_valid)
print('Validation accuracy: ' + str(val_acc*100) + '%')

start = time.time()
pred_test = np.zeros((len(testset_y), size[0], size[1], 1), dtype=np.int32)
for j in range(len(testset_y)//model.batch_size):
    pred_test[j*model.batch_size:(j+1)*model.batch_size] = \
        sess.run(model.pred, feed_dict={model.x: testset_x[j*model.batch_size:(j+1)*model.batch_size]})
pred_test[1368:1377] = sess.run(model.pred, feed_dict={model.x: testset_x[1368:1377]})
end = time.time() - start
print('Time per sample %.2fms' % (end/1.377))

test_acc = dice(testset_y, pred_test)
print('Test accuracy on 128x128 images: ' + str(test_acc*100) + '%')

class_pred = np.load('results' + os.sep + 'predictions.npy')
pred_test_vc = np.zeros((len(testset_y), size[0], size[1], 1), dtype=np.int32)
for i in range(len(testset_y)):
    pred_test_vc[i] = pred_test[i]*class_pred[i]

test_acc_vc = dice(testset_y, pred_test_vc)
print('Test accuracy on 128x128 images with classifier: ' + str(test_acc_vc*100) + '%')

idx = np.where(class_pred == 1)[0]
plt.figure()
plt.subplot(231)
plt.imshow(test_input[idx[0]].reshape(size), cmap="bone")
plt.imshow(test_output[idx[0]].reshape(size), alpha=0.3, cmap="Reds")
plt.subplot(232)
plt.imshow(test_input[idx[3]].reshape(size), cmap="bone")
plt.imshow(test_output[idx[3]].reshape(size), alpha=0.3, cmap="Reds")
plt.subplot(233)
plt.imshow(test_input[idx[2]].reshape(size), cmap="bone")
plt.imshow(test_output[idx[2]].reshape(size), alpha=0.3, cmap="Reds")
plt.subplot(234)
plt.imshow(test_input[idx[0]].reshape(size), cmap="bone")
plt.imshow(pred_test_vc[idx[0]].reshape(size), alpha=0.3, cmap="Reds")
plt.subplot(235)
plt.imshow(test_input[idx[3]].reshape(size), cmap="bone")
plt.imshow(pred_test_vc[idx[3]].reshape(size), alpha=0.3, cmap="Reds")
plt.subplot(236)
plt.imshow(test_input[idx[2]].reshape(size), cmap="bone")
plt.imshow(pred_test_vc[idx[2]].reshape(size), alpha=0.3, cmap="Reds")
plt.suptitle('Segmentation examples on scaled images', fontsize=16)
plt.savefig(model.folder_name + os.sep + 'examples_128_v' + str(version) + '.png')
plt.close()

fpred = []
pred_test_vc = np.reshape(pred_test_vc, (-1, size[0], size[1]))
for i in range(len(pred_test_vc)):
    img = imresize(pred_test_vc[i], (im_height, im_width)) > 0.5
    fpred.append(img)
fpred = np.array(fpred, np.float32)

test_acc_rs = dice(test_output_og, fpred)
print('Test accuracy on 1024x1024 images: ' + str(test_acc_rs*100) + '%')

fpred = []
black = np.zeros((im_height, im_width))
kernel = np.ones((12, 16), np.float32)
pred_test_vc = np.reshape(pred_test_vc, (-1, size[0], size[1]))
for i in range(len(pred_test_vc)):
    img = imresize(pred_test_vc[i], (im_height, im_width)) > 0.5
    if np.sum(img) < 900:
        fpred.append(black)
    else:
        fpred.append(cv2.dilate(np.array(img, np.float32), kernel))
fpred = np.array(fpred, np.float32)

test_acc_rs_d = dice(test_output_og, fpred)
print('Test accuracy on 1024x1024 images with discard: ' + str(test_acc_rs_d*100) + '%')

plt.figure()
plt.subplot(231)
plt.imshow(test_input_og[idx[0]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(test_output_og[idx[0]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.subplot(232)
plt.imshow(test_input_og[idx[3]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(test_output_og[idx[3]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.subplot(233)
plt.imshow(test_input_og[idx[2]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(test_output_og[idx[2]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.subplot(234)
plt.imshow(test_input_og[idx[0]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(fpred[idx[0]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.subplot(235)
plt.imshow(test_input_og[idx[3]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(fpred[idx[3]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.subplot(236)
plt.imshow(test_input_og[idx[2]].reshape((im_height, im_width)), cmap="bone")
plt.imshow(fpred[idx[2]].reshape((im_height, im_width)), alpha=0.3, cmap="Reds")
plt.axis('off')
plt.suptitle('Segmentation examples on original images', fontsize=16)
plt.savefig(model.folder_name + os.sep + 'examples_1024_v' + str(version) + '.png')
plt.close()
