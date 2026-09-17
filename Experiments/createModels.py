import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import keras

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.float32)



def get_matrix(model, y_true, X_test, model_type):
    file_name = 'matrices_weak_learners.txt' if model_type == 0 else 'matrices_strong_learners.txt'
    f= open(file_name, 'a')

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    TP = FN = FP = TN = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i] and y_true[i] == 0.0):
            TP += 1
        elif (y_true[i] == y_pred[i] and y_true[i] == 1.0):
            TN += 1
        elif (y_true[i] == 0.0 and y_pred[i] == 1.0):
            FN += 1
        else:
            FP += 1

    f.write(f"{TP} {FN} {FP} {TN}\n")
    #TP, FN, FP, TN
    f.close()


def create_sets(X, y):
    rand = random.randint(1, 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    return X_train, X_test, y_train, y_test

def simple_model(input_shape, n_classes):
    model = Sequential([
        layers.Conv2D(256, (3, 3), input_shape=input_shape, activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='sigmoid')
    ])
    return model

def train_and_save_model(model, model_num, X_train, X_test, y_train, y_test, mode):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test))
    if mode == 0:
        model.save(f'badModel{model_num}.keras')
    elif mode == 1:
        model.save(f'goodModel{model_num}.keras')


num_models = 18
num_weak_learners = 11
def createWeakLearners():
    for i in range(1, num_weak_learners+1):
        model = simple_model((50, 50, 1), 1)
        X_train, X_test, y_train, y_test = create_sets(X, y)
        train_and_save_model(model, i, X_train, X_test, y_train, y_test, 0)
        get_matrix(model, y_test, X_test, 0)


def createStrongLearners():
    model = simple_model((50, 50, 1), 1)
    model.summary()
    X_train, X_test, y_train, y_test = create_sets(X, y)
    for i in range(1, num_models + 1):
        train_and_save_model(model, i, X_train, X_test, y_train, y_test, 1)
        model = keras.models.load_model(f"goodModel{i}.keras") 
        get_matrix(model, y_test, X_test, 1)

createWeakLearners()
createStrongLearners()