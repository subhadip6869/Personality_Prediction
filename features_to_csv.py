import os
import numpy as np
import pandas as pd
import copy
from package.features import *

# Initializing dataset directory
train_data_dir = f"{os.path.dirname(__file__)}/dataset/data1/training_set/"
test_data_dir = f"{os.path.dirname(__file__)}/dataset/data1/test_set/"

# Initializing images directories
train_data = {"image": [], "class": []}
for i in os.listdir(train_data_dir):
    for file in os.listdir(os.path.join(train_data_dir, i)):
        train_data["image"].append(os.path.join(train_data_dir, i, file).replace("\\", "/"))
        train_data["class"].append(i)
test_data = {"image": [], "class": []}
for i in os.listdir(test_data_dir):
    for file in os.listdir(os.path.join(test_data_dir, i)):
        test_data["image"].append(os.path.join(test_data_dir, i, file).replace("\\", "/"))
        test_data["class"].append(i)

# converting dictionaries into dataframe
train_data = pd.DataFrame(train_data)
train_data.head()
test_data = pd.DataFrame(test_data)
test_data.head()

# showing dataframe shapes
print(train_data.shape)
print(test_data.shape)

# extracting features from images and storing their labels into y_test and y_train
x_train = []
y_train = []
for i in range(train_data.shape[0]):
    iminfo = []
    datalen = str(len(str(train_data.shape[0])))
    print(("\rTrain data processed: {:" + datalen + "d}/{:" + datalen + "d}").format(i+1, train_data.shape[0]), end="")
    iminfo.append(get_letter_slant(image_path=train_data.image.tolist()[i]))
    iminfo.append(get_line_slant(image_path=train_data.image.tolist()[i])[0])
    iminfo.append(get_margin_slope(image_path=train_data.image.tolist()[i]))
    iminfo.append(get_letter_size(image_path=train_data.image.tolist()[i])[0])
    iminfo.append(gap_between_words(image_path=train_data.image.tolist()[i])[0])
    x_train.append(iminfo)
    y_train.append([train_data["class"].tolist()[i]])
print()
x_test = []
y_test = []
for i in range(test_data.shape[0]):
    iminfo = []
    datalen = str(len(str(test_data.shape[0])))
    print(("\rTest data processed: {:" + datalen + "d}/{:" + datalen + "d}").format(i+1, test_data.shape[0]), end="")
    iminfo.append(get_letter_slant(image_path=test_data.image.tolist()[i]))
    iminfo.append(get_line_slant(image_path=test_data.image.tolist()[i])[0])
    iminfo.append(get_margin_slope(image_path=test_data.image.tolist()[i]))
    iminfo.append(get_letter_size(image_path=test_data.image.tolist()[i])[0])
    iminfo.append(gap_between_words(image_path=test_data.image.tolist()[i])[0])
    x_test.append(iminfo)
    y_test.append([test_data["class"].tolist()[i]])
print()


# converting train features and test labels into dictionaries
traindataset = {'letter_slant': [],
                'line_slant': [],
                'margin_slope': [],
                'letter_size': [],
                'word_spacing': [],
                'personality': []}
# testdataset = copy.deepcopy(traindataset)

for i in range(len(x_train)):
    traindataset['letter_slant'].append(x_train[i][0])
    traindataset['line_slant'].append(x_train[i][1])
    traindataset['margin_slope'].append(x_train[i][2])
    traindataset['letter_size'].append(x_train[i][3])
    traindataset['word_spacing'].append(x_train[i][4])
    traindataset['personality'].append(y_train[i][0])

for i in range(len(x_test)):
    traindataset['letter_slant'].append(x_test[i][0])
    traindataset['line_slant'].append(x_test[i][1])
    traindataset['margin_slope'].append(x_test[i][2])
    traindataset['letter_size'].append(x_test[i][3])
    traindataset['word_spacing'].append(x_test[i][4])
    traindataset['personality'].append(y_test[i][0])

# converting features into dataframe
traindataset = pd.DataFrame(traindataset)
# testdataset = pd.DataFrame(testdataset)

# saving dataframes as csv file
traindataset.to_csv(f"{os.path.dirname(__file__)}/dataset/features.csv", index=False)
# testdataset.to_csv(f"{os.path.dirname(__file__)}/dataset/test_features.csv", index=False)