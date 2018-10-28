import json
import pickle
import os
from data_genarator import DataGenerator
import sys
import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense, BatchNormalization, Dropout, AveragePooling1D, ReLU, \
    GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import plot_model
import pandas as pd
from keras.optimizers import SGD

# List of drivers
ALL_DRIVERS = ['ID0',
               'ID1',
               'ID2',
               'ID3',
               'ID4',
               'ID5',
               'ID6',
               'ID7',
               'ID8',
               'ID9',
               'ID10',
               'ID11',
               'ID12',
               'ID13',
               'ID14',
               'ID15',
               'ID16',
               'ID17',
               'ID18',
               'ID19',
               'ID20',
               'ID21',
               'ID22',
               'ID23',
               'ID24',
               'ID25',
               'ID26',
               'ID27',
               'ID28',
               'ID29',
               'ID30',
               'ID31',
               'ID32',
               'ID33']
# List of categories
CATEGORIES = []
TARGETS = []
modes = {'gender', 'age', 'experience', '1vsall', 'allvsall'}

# DO NOT CHANGE (length of a single slot in seconds, car dependant):
TIME = 0.05

SEG_LEN = 10  # sec, length of sample (window)
WINDOW_SIZE = int(SEG_LEN / TIME)

BATCH_SIZE = 32
CHANNELS = 1
SENSOR = ""
EPOCHS = 3
BINARY = 0

LOAD_MODEL = True


def make_local_conv(input_length, name):
    local_filter_size = 8
    local_kernel_size = 5
    local_stride = 1

    inp1 = keras.layers.Input(shape=(input_length, 1), dtype='float32', name=name)

    normalize1 = BatchNormalization()(inp1)
    local_conv1 = Conv1D(filters=local_filter_size, kernel_size=local_kernel_size, strides=local_stride,
                         padding='causal', activation='relu')(normalize1)
    local_pool1 = MaxPool1D(pool_size=2)(local_conv1)
    local_conv1 = Conv1D(filters=local_filter_size * 2, kernel_size=local_kernel_size, strides=local_stride,
                         padding='causal', activation='relu')(local_pool1)
    local_pool1 = MaxPool1D(pool_size=2)(local_conv1)
    local_conv1 = Conv1D(filters=local_filter_size * 4, kernel_size=local_kernel_size, strides=local_stride,
                         padding='causal', activation='relu')(local_pool1)

    return inp1, MaxPool1D(pool_size=2)(local_conv1)


# This is the classifier
def define_model(driver_num):
    if SENSOR == 'sensors':
        inputs = []
        local_pools = []
        for i in range(CHANNELS):
            inp, local_pool = make_local_conv(WINDOW_SIZE, 'inp' + str(i))
            inputs.append(inp)
            local_pools.append(local_pool)

        concat = keras.layers.concatenate(local_pools)
        flatten = Flatten()(concat)
    else:
        inputs, local_pool = make_local_conv(WINDOW_SIZE, 'inp1')
        flatten = Flatten()(local_pool)

    dense = Dense(128, activation='relu')(flatten)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation='relu')(dense)

    if BINARY:
        dense = Dense(1, activation='sigmoid')(dense)
    else:
        dense = Dense(driver_num, activation='softmax')(dense)

    mdl = Model(inputs=inputs, outputs=dense)

    return mdl


def set_categories():
    df = pd.read_csv('name_to_attr.csv')
    categories = df[MODE].unique()
    print("Categories: {0}".format(categories))
    return categories


# Entry point
if __name__ == "__main__":
    MODE = sys.argv[1]
    if MODE not in modes:
        sys.stderr("Requested mode not valid.")
        sys.exit(-1)

    FILE_ID = MODE

    if MODE == '1vsall':
        CATEGORIES = ALL_DRIVERS
        TARGETS.append(sys.argv[2])
        SENSOR = sys.argv[3]
        EPOCHS = 2
        CHANNELS = 4
        BINARY = 1
        FILE_ID = FILE_ID + '_' + TARGETS[0] + '_' + SENSOR
    elif MODE == 'allvsall':
        CATEGORIES = ALL_DRIVERS
        TARGETS = ALL_DRIVERS
        SENSOR = sys.argv[2]
        EPOCHS = 2
        CHANNELS = 1
        BINARY = 0
        FILE_ID = FILE_ID + '_' + SENSOR
    elif MODE == 'gender':
        CATEGORIES = set_categories()
        TARGETS.append(0)
        CHANNELS = int(sys.argv[2])
        if CHANNELS == 1:
            SENSOR = sys.argv[3]
        EPOCHS = 4
        BINARY = 1
        FILE_ID = FILE_ID + '_' + str(CHANNELS) + '_' + SENSOR
    elif MODE == 'age':
        CATEGORIES = set_categories()
        TARGETS = CATEGORIES
        CHANNELS = int(sys.argv[2])
        if CHANNELS == 1:
            SENSOR = sys.argv[3]
        EPOCHS = 4
        BINARY = 0
        FILE_ID = FILE_ID + '_' + str(CHANNELS) + '_' + SENSOR
    elif MODE == 'experience':
        CATEGORIES = set_categories()
        TARGETS = CATEGORIES
        CHANNELS = int(sys.argv[2])
        if CHANNELS == 1:
            SENSOR = sys.argv[3]
        EPOCHS = 4
        BINARY = 0
        FILE_ID = FILE_ID + '_' + str(CHANNELS) + '_' + SENSOR

    # Datasets
    with open('partitions2/' + MODE + '_partition.pkl', "rb") as read_file:
        partition = pickle.load(read_file)
    print("Partition loaded.")
    print("Train samples: {0}".format(len(partition['train'])))
    print("Test samples: {0}".format(len(partition['test'])))
    print("Validation samples: {0}".format(len(partition['validation'])))

    if MODE == '1vsall' or MODE == 'allvsall':
        with open("label_dict.json", "r") as read_file:
            labels = json.load(read_file)
    else:
        with open("label_dict_" + MODE + ".json", "r") as read_file:
            labels = json.load(read_file)
    print("Label dictionary loaded.")

    print("Total samples: {0}".format(len(labels)))
    training_generator = DataGenerator(partition['train'], labels, TARGETS, CATEGORIES, BINARY, CHANNELS, "train",
                                       sensor=SENSOR,
                                       batch_size=BATCH_SIZE)
    testing_generator = DataGenerator(partition['test'], labels, TARGETS, CATEGORIES, BINARY, CHANNELS, "test",
                                      sensor=SENSOR,
                                      batch_size=BATCH_SIZE)
    validation_generator = DataGenerator(partition['validation'], labels, TARGETS, CATEGORIES, BINARY, CHANNELS, "test",
                                         sensor=SENSOR,
                                         batch_size=BATCH_SIZE)

    model = define_model(len(CATEGORIES))

    # Stop training when a monitored quantity has stopped improving.
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')

    # Reduce learning rate when a metric has stopped improving.
    # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    # This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto')

    callbacks = [earlyStopping, reduce_lr_loss]

    print(model.summary())

    # compile model
    if BINARY:
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        epochs=EPOCHS,
                        workers=8)
    # plot_model(model, to_file='model_multichannel.png')
    score, acc = model.evaluate_generator(generator=testing_generator)
    print(model.metrics_names)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Number of categories:', len(CATEGORIES))

    ### SERIALIZE MODEL ###
    #######################
    if not os.path.exists('MODELS/'):
        os.makedirs('MODELS/')
    model.save("MODELS/" + FILE_ID + ".h5")
    #######################

    ### DOCUMENT RESULTS IN A FILE ###
    ##################################
    if not os.path.exists('RESULTS/'):
        os.makedirs('RESULTS/' + FILE_ID)

    with open('RESULTS/' + FILE_ID + '_results.txt', 'a') as file:
        file.write("\n{0},{1}".format(FILE_ID, acc))
    file.close()
    ##################################
