#Importing necessary libraries
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier

# initialise the dataset as given in the question.
data = {'id':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'x1':[-5.86, -10.97, 0.79, -0.59, 3.63, 2.02, -6.41, 6.13, -2.35, 2.66, -3.71, 2.4],
        'x2':[-2.0, -1.0, -2.0, 1.0, -2.0, -5.0, -1.0, -7.0, 6.0, -3.0, 2.0, 1.0],
        'y':[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]}
 
# Create DataFrame
df = pd.DataFrame(data)

X = df[['x1', 'x2']]
y = df[['y']]

#Question 1. a.

#Creating K folds
kfolds = y.size

#Store the cross validation errors to calculate their average
crossValidationError = []

#Run the loop for all the rows in the dataset to implement LEAVE ONE OUT CROSS VALIDATION
for i in range(kfolds):
  #Create training and testing data for each fold
  rowsTest = []
  rowsTrain = []
  for row in df.itertuples(index=True, name='Pandas'):
    if(row.Index == i):
      rowsTest.append([row.x1, row.x2, row.y])
    else:
      rowsTrain.append([row.x1, row.x2, row.y])
  testData = pd.DataFrame(rowsTest, columns=["x1", "x2", "y"])
  trainData = pd.DataFrame(rowsTrain, columns=["x1", "x2", "y"])
  
  #Creating traing and testing splits
  X_train = trainData[['x1', 'x2']]
  X_test = testData[['x1', 'x2']]
  y_train = trainData[['y']]
  y_test = testData[['y']]

  #KNN algorithm with nearest neighbors as 1 and manhattan distance metric
  classifier = KNeighborsClassifier(n_neighbors=1, metric="manhattan")
  classifier.fit(X_train, y_train.values.ravel())
  y_pred = classifier.predict(X_test)

  #Print the predicted values for all the k folds
  print("Index : ", i)
  print(y_pred)
  print(y_test.values.ravel())

  #Calculate the error
  summation = 0  
  n = len(y_test) 
  for i in range (0,n):  
    difference = y_test.to_numpy()[i] - y_pred[i]  
    squared_difference = difference**2  
    summation = summation + squared_difference  
  MSE = summation/n  
  print ("Error is: " , MSE)
  crossValidationError.append(MSE)
  print("------------------------")  

print(crossValidationError)
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

print("Total Error is : ", Average(crossValidationError))

#Question 1. b.

#Creating Manhattan Distance Metric

class distanceMetrics:
  def __init__(self):
    pass
  def manhattanDistance(self, vector1, vector2):
    self.vectorA, self.vectorB = vector1, vector2
    if len(self.vectorA) != len(self.vectorB):
      raise ValueError("Unequal Lengths of both the vectors")
    return np.abs(np.array(self.vectorA) - np.array(self.vectorB)).sum()

#Calculate the nearest neighbors

#For point 3
distance = []

predictionTuple = list(X.itertuples(index=False, name=None))[2]

for row in df.itertuples(index=True, name='Pandas'):
  currentTuple = (row.x1, row.x2)
  resultTuple = (row.Index + 1, distanceMetrics().manhattanDistance(predictionTuple, currentTuple))
  distance.append(resultTuple)

distance.sort(key = lambda x: x[1])
print("3 nearest neighbors for index 3 with their respective distances are : (Index, distance)")
print(distance[1:4])

#Calculate the nearest neighbors

#For point 10
distance = []

predictionTuple = list(X.itertuples(index=False, name=None))[9]

for row in df.itertuples(index=True, name='Pandas'):
  currentTuple = (row.x1, row.x2)
  resultTuple = (row.Index + 1, distanceMetrics().manhattanDistance(predictionTuple, currentTuple))
  distance.append(resultTuple)

distance.sort(key = lambda x: x[1])
print("3 nearest neighbors for index 3 with their respective distances are : (Index, distance)")
print(distance[1:4])

#Question 1. c.

#Creating K folds
kfolds = 3

#Rule for the testing dataset
#ID mod 3 = i − 1

crossValidationError = []

for i in range(kfolds):
  #Create training and testing data for each fold
  rowsTest = []
  rowsTrain = []
  for row in df.itertuples(index=True, name='Pandas'):
    if((row.Index + 1)%3 == i):
      rowsTest.append([row.x1, row.x2, row.y])
    else:
      rowsTrain.append([row.x1, row.x2, row.y])

  testData = pd.DataFrame(rowsTest, columns=["x1", "x2", "y"])
  trainData = pd.DataFrame(rowsTrain, columns=["x1", "x2", "y"])
  
  #Creating traing and testing splits
  X_train = trainData[['x1', 'x2']]
  X_test = testData[['x1', 'x2']]
  y_train = trainData[['y']]
  y_test = testData[['y']]

  #KNN algorithm
  classifier = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
  classifier.fit(X_train, y_train.values.ravel())
  y_pred = classifier.predict(X_test)

  print("Index : ", i)
  print(y_pred)
  print(y_test.values.ravel())

  summation = 0  
  n = len(y_test) 
  for i in range (0,n):  
    difference = y_test.to_numpy()[i] - y_pred[i]  
    squared_difference = difference**2  
    summation = summation + squared_difference  
  MSE = summation/n  
  print ("Error is: " , MSE)
  crossValidationError.append(MSE)
  print("------------------------")  


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

print("Total Error is : ", Average(crossValidationError))
