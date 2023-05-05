import os
import numpy as np
import pandas as pd
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
    print("\rTrain data processed: {:3d}/{:3d}".format(i+1, train_data.shape[0]), end="")
    iminfo.append(get_letter_slant(image_path=train_data.image.tolist()[i]))
    iminfo.append(get_line_slant(image_path=train_data.image.tolist()[i])[0])
    iminfo.append(get_letter_size(image_path=train_data.image.tolist()[i])[0])
    iminfo.append(gap_between_words(image_path=train_data.image.tolist()[i])[0])
    x_train.append(iminfo)
    y_train.append([train_data["class"].tolist()[i]])
print()
x_test = []
y_test = []
for i in range(test_data.shape[0]):
    iminfo = []
    print("\rTest data processed: {:3d}/{:3d}".format(i+1, test_data.shape[0]), end="")
    iminfo.append(get_letter_slant(image_path=test_data.image.tolist()[i]))
    iminfo.append(get_line_slant(image_path=test_data.image.tolist()[i])[0])
    iminfo.append(get_letter_size(image_path=test_data.image.tolist()[i])[0])
    iminfo.append(gap_between_words(image_path=test_data.image.tolist()[i])[0])
    x_test.append(iminfo)
    y_test.append([test_data["class"].tolist()[i]])


# converting train features and test labels into dictionaries
traindataset = {}
for i in range(x_train.shape[0]):
    traindataset['letter_slant'] = [data[0] for data in x_train]
    traindataset['line_slant'] = [data[1] for data in x_train]
    traindataset['letter_size'] = [data[2] for data in x_train]
    traindataset['word_spacing'] = [data[3] for data in x_train]
    traindataset['personality'] = [data[0] for data in y_train]
testdataset = {}
for i in range(x_test.shape[0]):
    testdataset['letter_slant'] = [data[0] for data in x_test]
    testdataset['line_slant'] = [data[1] for data in x_test]
    testdataset['letter_size'] = [data[2] for data in x_test]
    testdataset['word_spacing'] = [data[3] for data in x_test]
    testdataset['personality'] = [data[0] for data in y_test]

# converting features into dataframe
traindataset = pd.DataFrame(traindataset)
testdataset = pd.DataFrame(testdataset)

# saving dataframes as csv file
traindataset.to_csv(f"{os.path.dirname(__file__)}/dataset/train_features1.csv", index=False)
testdataset.to_csv(f"{os.path.dirname(__file__)}/dataset/test_data1.csv", index=False)