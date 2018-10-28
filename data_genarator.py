import numpy as np
import keras
import h5py
import pywt
import json

# DO NOT CHANGE (length of a single slot in seconds, car dependant):
TIME = 0.05

SEG_LEN = 10  # sec, length of sample (window)
WINDOW_SIZE = int(SEG_LEN / TIME)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_ids, labels, targets, categories, binary, n_channels, type, sensor='clutch', batch_size=32, dim=WINDOW_SIZE):
        'Initialization'
        self.targets = targets
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.on_epoch_end()
        self.list_of_sensors = []
        self.inp_names = []
        self.type = type
        self.binary = binary
        self.categories = categories

        if self.binary:
            self.n_classes = 2
        else:
            self.n_classes = len(targets)

        with open('label_dict.json', 'r') as f:
            self.label_dict_name = json.load(f)

        if self.n_channels == 1:
            if sensor!= "":
                self.list_of_sensors = [sensor]
            else:
                self.list_of_sensors = ['clutch']
            self.inp_names = ['inp1']
        else:
            self.list_of_sensors = ['clutch', 'gaspedal', 'rpm', 'speed']
            self.inp_names = ['inp0', 'inp1', 'inp2', 'inp3']

        print(
            "Created data generator with {0} number of channels, {1} targets, labels_dict of length {2}, number of "
            "classes {3}, list of ids with length {4}".format(self.n_channels, self.targets, len(self.labels),
                                                              self.n_classes, len(self.list_IDs)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def reshape_helper(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        if self.n_classes == 2:
            y = np.empty(self.batch_size, dtype=int)
        else:
            y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Map labels to integers
        if self.binary:
            driver2id = dict((t, int(t in self.targets)) for t in set(self.categories))
        else:
            driver2id = dict(
                (t, keras.utils.to_categorical(i, num_classes=self.n_classes)) for i, t in enumerate(self.categories))

        if self.type == "train":
            h5f = h5py.File("sliced_data/dset1.h5", 'r')
        elif self.type == "test":
            h5f = h5py.File("sliced_data/dset2.h5", 'r')

        X_dict = {}
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            for sensor_, j in zip(self.list_of_sensors, range(self.n_channels)):
                x = h5f[sensor_][int(ID)]
                X[i, :, j] = x
            if self.binary:
                y[i] = driver2id[self.labels[str(ID)]]
            else:
                y[i] = np.array(driver2id[self.labels[str(ID)]])

        for inp, i in zip(self.inp_names, range(self.n_channels)):
            X_dict[inp] = self.reshape_helper(X[:, :, i])
        h5f.close()
        return X_dict, y
