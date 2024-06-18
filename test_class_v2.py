import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from classificator import Model

size = (128, 128)
name = 'class_v4'
version = 44
tresh = 0.2692
classes = ['0', '1']

sess = tf.InteractiveSession()
model = Model(name=name, test=True)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(model.folder_name)
saver.restore(sess, ckpt.model_checkpoint_path)


def plot_confusion_matrix(cm, classes, nameplot,
                          normalize=False,
                          cmap=plt.cm.Blues):

    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix [%]'
    else:
        title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(model.folder_name + os.sep + nameplot + str(version) + '.png')
    plt.close()

print('Loading data...', end=' ')
valid_input = np.load('data' + os.sep + 'valid_input_' + str(size[0]) + '.npy')
validset_y = np.load('data' + os.sep + 'valid_class_' + str(size[0]) + '.npy')
valid_gender = np.load('data' + os.sep + 'valid_gender_' + str(size[0]) + '.npy')

images = valid_input.astype(np.float32) * (1. / 255) - 0.5
validset_x = np.reshape(images, (-1, size[0], size[1], 1))

test_input = np.load('data' + os.sep + 'test_input_' + str(size[0]) + '.npy')
testset_y = np.load('data' + os.sep + 'test_class' + '.npy')
test_gender = np.load('data' + os.sep + 'test_gender_' + str(size[0]) + '.npy')
test_id = np.load('data' + os.sep + 'test_id_' + str(size[0]) + '.npy')

test_input_og = np.load('data' + os.sep + 'test_input_' + str(1024) + '.npy')
test_output_og = np.load('data' + os.sep + 'test_output_' + str(1024) + '.npy')

test_input = test_input.astype(np.float32) * (1. / 255) - 0.5
testset_x = np.reshape(test_input, (-1, size[0], size[1], 1))

print('Done.')

pred = np.zeros((len(validset_y), ), dtype=np.int32)
for j in range(len(validset_y)//model.batch_size):
    softmax = sess.run(model.sm, feed_dict={model.x: validset_x[j*model.batch_size:(j+1)*model.batch_size]})
    pred[j*model.batch_size:(j+1)*model.batch_size] = np.array([softmax[i][1] > tresh
                                                                for i in range(model.batch_size)])

cm = confusion_matrix(validset_y, pred)
plot_confusion_matrix(cm, classes=classes, nameplot='cm_valid')
plot_confusion_matrix(cm, classes=classes, normalize=True, nameplot='cm_valid_norm')

pred = np.zeros((len(testset_y), ), dtype=np.int32)
for j in range(len(testset_y)//model.batch_size):
    softmax = sess.run(model.sm, feed_dict={model.x: testset_x[j*model.batch_size:(j+1)*model.batch_size]})
    pred[j*model.batch_size:(j+1)*model.batch_size] = np.array([softmax[i][1] > tresh
                                                                for i in range(model.batch_size)])

softmax = sess.run(model.sm, feed_dict={model.x: testset_x[1368:1377]})
pred[1368:1377] = np.array([softmax[i][1] > tresh for i in range(9)])

cm = confusion_matrix(testset_y, pred)
plot_confusion_matrix(cm, classes=classes, nameplot='cm_test')
plot_confusion_matrix(cm, classes=classes, normalize=True, nameplot='cm_test_norm')
np.save(model.folder_name + os.sep + 'predictions', pred)

tp = np.where(pred == testset_y)[0]
fp = np.where(pred != testset_y)[0]

idx = 0
while testset_y[fp[idx]] == 0:
    idx += 1

plt.figure()
plt.subplot(121)
plt.imshow(test_input_og[tp[0]], cmap="bone")
plt.imshow(test_output_og[tp[0]], alpha=0.3, cmap="Reds")
plt.subplot(122)
plt.imshow(test_input_og[tp[1]], cmap="bone")
plt.imshow(test_output_og[tp[1]], alpha=0.3, cmap="Reds")
plt.suptitle('Correctly classified images', fontsize=16)
plt.savefig(model.folder_name + os.sep + 'correct_classes.png')
plt.close()

plt.figure()
plt.subplot(121)
plt.imshow(test_input_og[fp[idx]], cmap="bone")
plt.imshow(test_output_og[fp[idx]], alpha=0.3, cmap="Reds")
plt.subplot(122)
plt.imshow(test_input_og[fp[1]], cmap="bone")
plt.imshow(test_output_og[fp[1]], alpha=0.3, cmap="Reds")
plt.suptitle('Incorrectly classified images', fontsize=16)
plt.savefig(model.folder_name + os.sep + 'wrong_classes.png')
plt.close()
