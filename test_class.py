import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from classificator import Model

size = (128, 128)
version = 48
name = 'class_v4'

sess = tf.InteractiveSession()
model = Model(name=name, test=True)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(model.folder_name)
saver.restore(sess, ckpt.model_checkpoint_path)

print('Loading data...', end=' ')
valid_input = np.load('data' + os.sep + 'valid_input_' + str(size[0]) + '.npy')
valid_output = np.load('data' + os.sep + 'valid_class_' + str(size[0]) + '.npy')
valid_gender = np.load('data' + os.sep + 'valid_gender_' + str(size[0]) + '.npy')
valid_input_aug = np.load('data' + os.sep + 'valid_input_augmented_' + str(size[0]) + '.npy')
valid_output_aug = np.load('data' + os.sep + 'valid_class_' + str(size[0]) + '.npy')

images = np.vstack((valid_input, valid_input_aug))
classes = np.hstack((valid_output, valid_output_aug))
images = images.astype(np.float32) * (1. / 255) - 0.5

validset_x = np.reshape(images, (-1, size[0], size[1], 1))
validset_y = classes

print('Done.')

tresh = np.array([0.268, 0.269, 0.27, 0.275, 0.28, 0.29, 0.3, 0.4, 0.5, 0.6, 0.7, 0.72, 0.74])
pred = np.zeros((len(validset_y), ), dtype=np.int32)
roc = np.zeros((len(tresh), 4))
sens = np.zeros((len(tresh), ))
spec = np.zeros((len(tresh), ))
for t in range(len(tresh)):
    for j in range(len(validset_y)//model.batch_size):
        softmax = sess.run(model.sm, feed_dict={model.x: validset_x[j*model.batch_size:(j+1)*model.batch_size]})
        pred[j*model.batch_size:(j+1)*model.batch_size] = np.array([softmax[i][1] > tresh[t]
                                                                    for i in range(model.batch_size)])
    roc[t][0] = np.count_nonzero(pred[pred == validset_y] == 1)  # TP
    roc[t][1] = np.count_nonzero(pred[pred == validset_y] == 0)  # TN
    roc[t][2] = np.count_nonzero(pred[pred != validset_y] == 1)  # FP
    roc[t][3] = np.count_nonzero(pred[pred != validset_y] == 0)  # FN

    sens[t] = roc[t][0] / (roc[t][0] + roc[t][3])
    spec[t] = roc[t][1] / (roc[t][1] + roc[t][2])

plt.figure()
plt.plot(1-spec, sens)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title("ROC curve")
plt.savefig(model.folder_name + os.sep + 'roc' + str(version) + '.png')
plt.close()

plt.figure()
plt.plot(tresh, [roc[i][0] for i in range(len(tresh))])
plt.xlabel('Treshold')
plt.ylabel('True Positive')
plt.savefig(model.folder_name + os.sep + 'tp' + str(version) + '.png')
plt.close()

plt.figure()
plt.plot(tresh, [roc[i][1] for i in range(len(tresh))])
plt.xlabel('Treshold')
plt.ylabel('Ture negative')
plt.savefig(model.folder_name + os.sep + 'tn' + str(version) + '.png')
plt.close()

plt.figure()
plt.plot(tresh, [roc[i][2] for i in range(len(tresh))])
plt.xlabel('Treshold')
plt.ylabel('False positive')
plt.savefig(model.folder_name + os.sep + 'fp' + str(version) + '.png')
plt.close()

plt.figure()
plt.plot(tresh, [roc[i][3] for i in range(len(tresh))])
plt.xlabel('Treshold')
plt.ylabel('False negative')
plt.savefig(model.folder_name + os.sep + 'fn' + str(version) + '.png')
plt.close()

print('Model ' + str(version))
print('Tresh ' + str(tresh[1]) + ': sens:' + str(sens[1]) + ', spec:' + str(spec[1]))
print('Tresh ' + str(tresh[2]) + ': sens:' + str(sens[2]) + ', spec:' + str(spec[2]))
print('Tresh ' + str(tresh[3]) + ': sens:' + str(sens[3]) + ', spec:' + str(spec[3]))
print('Tresh ' + str(tresh[4]) + ': sens:' + str(sens[4]) + ', spec:' + str(spec[4]))
