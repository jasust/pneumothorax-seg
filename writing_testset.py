import os
import cv2
import glob
import time
import pydicom
import numpy as np
import pandas as pd


###  FUNKCIJE  ###
def resize(img, new_dim):
    img2 = cv2.resize(img, (new_dim, new_dim))
    return img2


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


###  UCITAVANJE  ###
test_glob = os.path.join('..', 'dataset', 'dicom-images-test', '*', '*', '*.dcm')
label_glob = os.path.join('..', 'dataset', 'stage_2_train.csv')

start = time.time()

test_fns = sorted(glob.glob(test_glob))
df_full = pd.read_csv(label_glob, index_col='ImageId')
print('Testing data   ({} samples)'.format(len(test_fns)))

end = time.time() - start
print('Loading data finished... (%.2fs)' % end)

###  ISPISIVANJE PODATAKA  ###
im_height = 1024
im_width = 1024
size = 128

start = time.time()
print('Writing test data...')

Y_test_r = np.zeros((len(test_fns), size, size), dtype=np.uint8)
Y_test = np.zeros((len(test_fns), im_height, im_width), dtype=np.uint8)
X_test = np.zeros((len(test_fns), im_height, im_width), dtype=np.uint8)
B_test = np.zeros((len(test_fns)), dtype=np.int32)

cnt = 0
for n, _id in enumerate(test_fns):
    dataset = pydicom.read_file(_id)
    X_test[n] = dataset.pixel_array

    try:
        if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], 'EncodedPixels']:
            Y_test[n] = np.zeros((im_height, im_width))
            Y_test_r[n] = np.zeros((size, size))
            B_test[n] = 0
        else:
            if type(df_full.loc[_id.split(os.sep)[-1][:-4], 'EncodedPixels']) == str:
                Y = rle2mask(df_full.loc[_id.split(os.sep)[-1][:-4], 'EncodedPixels'], im_height, im_width)
            else:
                Y = np.zeros((im_height, im_width))
                for x in df_full.loc[_id.split(os.sep)[-1][:-4], 'EncodedPixels']:
                    Y = Y + rle2mask(x, im_height, im_width)

            Y_test[n] = Y
            Y_test_r[n] = resize(Y, size)
            B_test[n] = 1
    except KeyError:
        Y_test[n] = np.zeros((im_height, im_width))
        Y_test_r[n] = np.zeros((size, size))
        B_test[n] = 0

dirname = 'data' + os.sep
np.save(dirname + 'test_input_' + str(im_height), X_test)
np.save(dirname + 'test_output_' + str(im_height), Y_test)
np.save(dirname + 'test_output_' + str(size), Y_test_r)
np.save(dirname + 'test_class', B_test)

end = time.time() - start
print('Writing test data finished... (%.2fs)' % end)

print('Done!')
