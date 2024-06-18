import os
import cv2
import glob
import time
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###  PARAMETRI  ###
size = 128
AUGMENT = True
DEBUG = False
SHOW = False
train_size = 9344


###  FUNKCIJE  ###
def show_dcm_info(dcm_file):

    print("Patient id..........:", dcm_file.PatientID)
    print("Patient's Age.......:", dcm_file.PatientAge)
    print("Patient's Sex.......:", dcm_file.PatientSex)
    print("Modality............:", dcm_file.Modality)
    print("Body Part Examined..:", dcm_file.BodyPartExamined)
    print("View Position.......:", dcm_file.ViewPosition)

    if 'PixelData' in dcm_file:
        rows = int(dcm_file.Rows)
        cols = int(dcm_file.Columns)
        print("Image size..........: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dcm_file.PixelData)))
        if 'PixelSpacing' in dcm_file:
            print("Pixel spacing.......:", dcm_file.PixelSpacing)


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


def plot_pixel_array(img, mask, title):
    plt.figure()
    plt.subplot(141)
    plt.title('Original image')
    plt.imshow(img, cmap="bone")
    plt.imshow(mask, alpha=0.3, cmap="Reds")
    plt.subplot(142)
    plt.title('Flipped image')
    plt.imshow(np.flip(img, 1), cmap="bone")
    plt.imshow(np.flip(mask, 1), alpha=0.3, cmap="Reds")
    plt.subplot(143)
    plt.title('Image resized to 128x128')
    plt.imshow(resize(img, 128), cmap="bone")
    plt.imshow(resize(mask, 128), alpha=0.3, cmap="Reds")
    plt.subplot(144)
    plt.title('Image resized to 256x256')
    plt.imshow(resize(img, 256), cmap="bone")
    plt.imshow(resize(mask, 256), alpha=0.3, cmap="Reds")
    plt.suptitle(title, fontsize=16)
    plt.show()


def resize(img, new_dim):
    img2 = cv2.resize(img, (new_dim, new_dim))
    return img2

###  UCITAVANJE  ###
train_glob = os.path.join('..', 'dataset', 'dicom-images-train', '*', '*', '*.dcm')
test_glob = os.path.join('..', 'dataset', 'dicom-images-test', '*', '*', '*.dcm')
label_glob = os.path.join('..', 'dataset', 'train-rle.csv')

start = time.time()

train_fns = sorted(glob.glob(train_glob))[:train_size]
valid_fns = sorted(glob.glob(train_glob))[train_size:]
test_fns = sorted(glob.glob(test_glob))
df_full = pd.read_csv(label_glob, index_col='ImageId')

total_size = len(train_fns)+len(valid_fns)+len(test_fns)
print('Training data  ({} samples):  {:.2f}%'.format(len(train_fns), 100*len(train_fns)/total_size))
print('Validation data ({} samples): {:.2f}%'.format(len(valid_fns), 100*len(valid_fns)/total_size))
print('Testing data   ({} samples):  {:.2f}%'.format(len(test_fns), 100*len(test_fns)/total_size))

end = time.time() - start
print('Loading data finished... (%.2fs)' % end)

##  PRIKAZIVANJE PODATAKA  ###
im_height = 1024
im_width = 1024

if SHOW:
    # Get validation images and masks
    X_valid = np.zeros((len(valid_fns), im_height, im_width), dtype=np.uint8)
    Y_valid = np.zeros((len(valid_fns), im_height, im_width), dtype=np.uint8)
    print('Showing sample images and masks ... ')

    num_empty = 2
    num_mask = 2

    for n, _id in enumerate(valid_fns):
        dataset = pydicom.read_file(_id)
        X_valid[n] = dataset.pixel_array

        try:
            if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                Y_valid[n] = np.zeros((im_height, im_width))

                if num_empty:
                    show_dcm_info(dataset)
                    plot_pixel_array(X_valid[n], Y_valid[n], 'No Pneumothorax Marker')
                    num_empty -= 1

            else:
                if type(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']) == str:
                    Y_valid[n] = rle2mask(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels'], im_height, im_width)
                else:
                    Y_valid[n] = np.zeros((im_height, im_width))
                    for x in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                        Y_valid[n] = Y_valid[n] + rle2mask(x, im_height, im_width)

                if num_mask:
                    show_dcm_info(dataset)
                    plot_pixel_array(X_valid[n], Y_valid[n], 'With Pneumothorax Marker')
                    num_mask -= 1
                else:
                    break

        except KeyError:
            Y_valid[n] = np.zeros((im_height, im_width))

###  PRIKAZIVANJE STATISTIKA  ###

if SHOW:
    train_stats = np.zeros((3, 10))
    valid_stats = np.zeros((3, 10))
    test_stats = np.zeros((3, 10))
    x_axis = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']

    print('Calculating training stats...')
    for n, _id in enumerate(train_fns):
        data_file = pydicom.read_file(_id)
        train_stats[1][int(data_file.PatientSex == 'M')] += 1
        train_stats[2][min(int(data_file.PatientAge) // 10, 9)] += 1
        try:
            if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                train_stats[0][0] += 1
            else:
                train_stats[0][1] += 1
        except KeyError:
            train_stats[0][0] += 1

    print('TRAIN STATS:')
    print('Mask: {}({:.2f}%) / NoMask: {}({:.2f}%)'.format(train_stats[0][1], train_stats[0][1]*100/len(train_fns),
                                                           train_stats[0][0], train_stats[0][0]*100/len(train_fns)))
    print('M: {}({:.2f}%) / F: {}({:.2f}%)'.format(train_stats[1][1], train_stats[1][1]*100/len(train_fns),
                                                   train_stats[1][0], train_stats[1][0]*100/len(train_fns)))
    plt.figure()
    plt.bar(x_axis, train_stats[2][:])
    plt.xlabel('Age')
    plt.ylabel('Number of Patients')
    plt.title('Age distribution for training set')
    plt.show()

    print('Calculating validation stats...')
    for n, _id in enumerate(valid_fns):
        data_file = pydicom.read_file(_id)
        valid_stats[1][int(data_file.PatientSex == 'M')] += 1
        valid_stats[2][min(int(data_file.PatientAge) // 10, 9)] += 1
        try:
            if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                valid_stats[0][0] += 1
            else:
                valid_stats[0][1] += 1
        except KeyError:
            valid_stats[0][0] += 1

    print('VALID STATS:')
    print('Mask: {}({:.2f}%) / NoMask: {}({:.2f}%)'.format(valid_stats[0][1], valid_stats[0][1]*100/len(valid_fns),
                                                           valid_stats[0][0], valid_stats[0][0]*100/len(valid_fns)))
    print('M: {}({:.2f}%) / F: {}({:.2f}%)'.format(valid_stats[1][1], valid_stats[1][1]*100/len(valid_fns),
                                                   valid_stats[1][0], valid_stats[1][0]*100/len(valid_fns)))
    plt.figure()
    plt.bar(x_axis, valid_stats[2][:])
    plt.xlabel('Age')
    plt.ylabel('Number of Patients')
    plt.title('Age distribution for validation set')
    plt.show()

    print('Calculating testing stats...')
    for n, _id in enumerate(test_fns):
        data_file = pydicom.read_file(_id)
        test_stats[0][int(data_file.PatientSex == 'M')] += 1
        test_stats[1][min(int(data_file.PatientAge) // 10, 9)] += 1

    print('TEST STATS:')
    print('M: {}({:.2f}%) / F: {}({:.2f}%)'.format(test_stats[0][1], test_stats[0][1]*100/len(test_fns),
                                                   test_stats[0][0], test_stats[0][0]*100/len(test_fns)))
    plt.figure()
    plt.bar(x_axis, test_stats[1][:])
    plt.xlabel('Age')
    plt.ylabel('Number of Patients')
    plt.title('Age distribution for test set')
    plt.show()

###  ISPISIVANJE PODATAKA  ###

start = time.time()
print('Writing training data...')

X_train = np.zeros((len(train_fns), size, size), dtype=np.uint8)
Y_train = np.zeros((len(train_fns), size, size), dtype=np.uint8)
X_train_a = np.zeros((len(train_fns), size, size), dtype=np.uint8)
Y_train_a = np.zeros((len(train_fns), size, size), dtype=np.uint8)
G_train = np.zeros((len(train_fns)), dtype=np.bool)
B_train = np.zeros((len(train_fns)), dtype=np.int32)

for n, _id in enumerate(train_fns):
    dataset = pydicom.read_file(_id)
    X_train[n] = resize(dataset.pixel_array, size)
    G_train[n] = dataset.PatientSex == 'M'

    try:
        if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
            Y_train[n] = np.zeros((size, size))
            B_train[n] = 0
        else:
            if type(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']) == str:
                Y = rle2mask(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels'], im_height, im_width)
            else:
                Y = np.zeros((im_height, im_width))
                for x in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                    Y = Y + rle2mask(x, im_height, im_width)
            Y_train[n] = resize(Y, size)
            B_train[n] = 1
    except KeyError:
        Y_train[n] = np.zeros((size, size))
        B_train[n] = 0

    if AUGMENT:
        X_train_a[n] = np.flip(X_train[n], 1)
        Y_train_a[n] = np.flip(Y_train[n], 1)

dirname = 'data' + os.sep
np.save(dirname + 'train_input_' + str(size), X_train)
np.save(dirname + 'train_output_' + str(size), Y_train)
np.save(dirname + 'train_gender_' + str(size), G_train)
np.save(dirname + 'train_class_' + str(size), B_train)
if AUGMENT:
    np.save(dirname + 'train_input_augmented_' + str(size), X_train_a)
    np.save(dirname + 'train_output_augmented_' + str(size), Y_train_a)

end = time.time() - start
print('Writing training data finished... (%.2fs)' % end)

start = time.time()
print('Writing validation data...')

X_valid = np.zeros((len(valid_fns), size, size), dtype=np.uint8)
Y_valid = np.zeros((len(valid_fns), size, size), dtype=np.uint8)
X_valid_a = np.zeros((len(valid_fns), size, size), dtype=np.uint8)
Y_valid_a = np.zeros((len(valid_fns), size, size), dtype=np.uint8)
G_valid = np.zeros((len(valid_fns)), dtype=np.bool)
B_valid = np.zeros((len(valid_fns)), dtype=np.int32)

for n, _id in enumerate(valid_fns):
    dataset = pydicom.read_file(_id)
    X_valid[n] = resize(dataset.pixel_array, size)
    G_valid[n] = dataset.PatientSex == 'M'

    try:
        if '-1' in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
            Y_valid[n] = np.zeros((size, size))
            B_valid[n] = 0
        else:
            if type(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']) == str:
                Y = rle2mask(df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels'], im_height, im_width)
            else:
                Y = np.zeros((im_height, im_width))
                for x in df_full.loc[_id.split(os.sep)[-1][:-4], ' EncodedPixels']:
                    Y = Y + rle2mask(x, im_height, im_width)
            Y_valid[n] = resize(Y, size)
            B_valid[n] = 1
    except KeyError:
        Y_valid[n] = np.zeros((size, size))
        B_valid[n] = 0

    if AUGMENT:
        X_valid_a[n] = np.flip(X_valid[n], 1)
        Y_valid_a[n] = np.flip(Y_valid[n], 1)

dirname = 'data' + os.sep
np.save(dirname + 'valid_input_' + str(size), X_valid)
np.save(dirname + 'valid_output_' + str(size), Y_valid)
np.save(dirname + 'valid_gender_' + str(size), G_valid)
np.save(dirname + 'valid_class_' + str(size), B_valid)
if AUGMENT:
    np.save(dirname + 'valid_input_augmented_' + str(size), X_valid_a)
    np.save(dirname + 'valid_output_augmented_' + str(size), Y_valid_a)

end = time.time() - start
print('Writing validation data finished... (%.2fs)' % end)

start = time.time()
print('Writing test data...')

X_test = np.zeros((len(test_fns), size, size), dtype=np.uint8)
G_test = np.zeros((len(test_fns)), dtype=np.bool)
Y_test = []

for n, _id in enumerate(test_fns):
    dataset = pydicom.read_file(_id)
    X_test[n] = resize(dataset.pixel_array, size)
    G_test[n] = dataset.PatientSex == 'M'
    Y_test.append(_id)

dirname = 'data' + os.sep
np.save(dirname + 'test_input_' + str(size), X_test)
np.save(dirname + 'test_gender_' + str(size), G_test)
np.save(dirname + 'test_id_' + str(size), np.array(Y_test))

end = time.time() - start
print('Writing test data finished... (%.2fs)' % end)

print('Done!')
