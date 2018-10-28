import pickle
import json
from math import ceil
import random
import sys
import numpy as np
import pandas as pd
from collections import Counter

NUM_OF_ENTRIES = 0

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

CATEGORIES = []
TARGETS = []
modes = {'gender', 'age', 'experience', '1vsall', 'allvsall'}

TIME = 0.05
SEG_LEN = 10 # sec, length of sample (window)
WINDOW_SHIFT = 3 # slots
WINDOW_SIZE = int(SEG_LEN / TIME)

TEST_DATA_RATIO = 0.2
VALIDATION_TRAIN_RATIO = 0.1


# We need the same number of samples from each class to have unbiased prediction
# list_of_ids - range(num_of_entries)
# label_dict - indexing this dictionary will give you back the label of the entry
# sample_nums - we have it serialized under sample_num.json
def get_partition(test_ids, train_ids, label_dict, sample_nums):
    random.seed(22)
    test_samples, train_samples, validation_samples = [], [], []
    test_samples_neg, train_samples_neg, validation_samples_neg = [], [], []

    ###
    # TRAIN DATA
    ###
    if MODE == '1vsall' or MODE == 'allvsall':
        min_sample_num = min([v for k, v in sample_nums.items() if k in CATEGORIES])
    else:
        min_sample_num = min([v for k, v in sample_nums.items() if int(k) in CATEGORIES])
    min_sample_num = int(min_sample_num * (1-TEST_DATA_RATIO))

    for category in CATEGORIES:
        id_samples = [str(id_) for id_ in train_ids if label_dict[str(id_)] == category]
        if category in TARGETS:
            train_samples.extend(random.sample(id_samples, min_sample_num))
        else:
            train_samples_neg.extend(random.sample(id_samples, min_sample_num))

    if BINARY:
        train_samples.extend(random.sample(train_samples_neg, len(train_samples)))
    else:
        train_samples.extend(train_samples_neg)

    random.shuffle(train_samples)

    ###
    # TEST/VALIDATION DATA
    ###
    if MODE == '1vsall' or MODE == 'allvsall':
        min_sample_num = min([v for k, v in sample_nums.items() if k in CATEGORIES])
    else:
        min_sample_num = min([v for k, v in sample_nums.items() if int(k) in CATEGORIES])
    min_sample_num = int(min_sample_num * TEST_DATA_RATIO)

    for category in CATEGORIES:
        id_samples = [str(id_) for id_ in test_ids if label_dict[str(id_)] == category]
        if category in TARGETS:
            test_samples.extend(random.sample(id_samples, min_sample_num))
            validation_samples.extend(random.sample(id_samples, int(min_sample_num*VALIDATION_TRAIN_RATIO)))
        else:
            test_samples_neg.extend(random.sample(id_samples, min_sample_num))
            validation_samples_neg.extend(random.sample(id_samples, int(min_sample_num * VALIDATION_TRAIN_RATIO)))

    if BINARY:
        test_samples.extend(random.sample(train_samples_neg, len(test_samples)))
        validation_samples.extend(random.sample(validation_samples_neg, len(validation_samples)))
    else:
        test_samples.extend(test_samples_neg)
        validation_samples.extend(validation_samples_neg)

    for id_ in validation_samples:
        if id_ in train_samples:
            train_samples.remove(id_)

    random.shuffle(test_samples)
    random.shuffle(validation_samples)

    partition = {}

    partition['train'] = train_samples
    partition['test'] = test_samples
    partition['validation'] = validation_samples

    print("Train sample length:", len(train_samples))
    print("Test sample length:", len(test_samples))
    print("Validation sample length:", len(validation_samples))

    return partition

def set_categories():
    df = pd.read_csv('name_to_attr.csv')
    categories = df[MODE].unique()
    print("Categories: {0}".format(categories))
    return categories

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
        BINARY = 1
        FILE_ID = FILE_ID + '_' + TARGETS[0] + '_' + SENSOR
    if MODE == 'allvsall':
        CATEGORIES = ALL_DRIVERS
        TARGETS = ALL_DRIVERS
        BINARY = 0
    if MODE == 'gender':
        CATEGORIES = set_categories()
        TARGETS.append(0)
        BINARY = 1
    if MODE == 'age':
        CATEGORIES = set_categories()
        print(CATEGORIES)
        TARGETS = CATEGORIES
        BINARY = 0
    if MODE == 'experience':
        CATEGORIES = set_categories()
        TARGETS = CATEGORIES
        BINARY = 0

    if MODE == '1vsall' or MODE == 'allvsall':
        with open("sample_num.json", "r") as read_file:
            sample_num = json.load(read_file)
    else:
        with open("sample_num_"+MODE+".json", "r") as read_file:
            sample_num = json.load(read_file)


    if MODE == '1vsall' or MODE == 'allvsall':
        with open("label_dict.json", "r") as read_file:
            label_dict = json.load(read_file)
    else:
        with open("label_dict_" + MODE + ".json", "r") as read_file:
            label_dict = json.load(read_file)
    print("Label dictionary loaded.")


    print("Making train/test/validation split...")
    print("TEST/ALL = ", TEST_DATA_RATIO)
    print("VALIDATION/TRAIN = ", VALIDATION_TRAIN_RATIO)
    ids = range(NUM_OF_ENTRIES)
    partition = get_partition(ids, ids, label_dict, sample_num)

    with open("partitions2/"+FILE_ID+"_partition.pkl", 'wb') as pickle_file:
        pickle.dump(partition, pickle_file)