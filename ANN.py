from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.models import Sequential
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Overriding the seed in os
import os
os.environ['PYTHONHASHSEED'] = '2021'
np.random.seed(2021)
tf.random.set_seed(2021)
random.seed(2021)

#import data
train_data = pd.read_csv('/content/ann/ann_train.csv')
test_data = pd.read_csv('/content/ann/ann_test.csv')
validation_data = pd.read_csv('/content/ann/ann_val.csv')

# Separating features and Class labels
y_train = train_data['Class']
X_train = train_data[:][:]
del X_train['Class']

y_test = test_data['Class']
X_test = test_data[:][:]
del X_test['Class']

y_val = validation_data['Class']
X_val = validation_data[:][:]
del X_val['Class']

# Artificial Neural Network Model
X = [4, 16, 32, 64]
training_accuracy = []
validation_accuracy = []

for i in X:
    print("Running for : " + str(i))

    # define the keras model
    model = Sequential()
    model.add(Dense(i, input_dim=60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=5, batch_size=10)

    # predicting for the validation data
    validation_pred = (model.predict(X_val) >= 0.5).astype(int)

    # predicting for the training data
    training_pred = (model.predict(X_train) >= 0.5).astype(int)

    # evaluate the keras model for validation data
    validationacc = accuracy_score(y_val, validation_pred)
    validation_accuracy.append(validationacc)

    # evaluate the keras model for training data
    validationtrain = accuracy_score(y_train, training_pred)
    training_accuracy.append(validationtrain)

# Printing validation and testing accuracies
print("Validation Accuracy :")
print(validation_accuracy)
print("Training Accuracy :")
print(training_accuracy)

# Plotting the graph between validationa and training accuracy
y1 = np.array(validation_accuracy)
y2 = np.array(training_accuracy)
x = np.array([4, 16, 32, 64])

plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.plot(x, y1, "-b", label="Validation Accuracy")
plt.plot(x, y2, "-r", label="Training Accuracy")
plt.legend(loc="lower right")
plt.show()

# ANN for the testing data
X = [32]

test_acc = []

for i in X:
    print("Running for : " + str(i))

    # define the keras model
    model = Sequential()
    model.add(Dense(i, input_dim=60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=5, batch_size=10)

    # predicting value for testing data
    testing_pred = (model.predict(X_test) >= 0.5).astype(int)

    # evaluate the keras model for testing data
    testingtrain = accuracy_score(y_test, testing_pred)
    test_acc.append(testingtrain)

print("Testing accuracy is : ")
print(test_acc)
