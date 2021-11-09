# The distance between P and the 398 data points using the following distance measures: 
#1) Euclidean distance, 2) Manhattan block metric, 3)Minkowski metric (for power=7), 4) Chebyshev distance, and 5) Cosine distance.
#List the closest 6 points for each distance.
import scipy
from scipy.spatial import distance
s = []   # pointer to store the co-ordinates of the displacement and weight

#creating empty lists to store the distances for each distance type
eu_distances = []
man_distances = []
min_distances =[]
cheby_distances = []
cosine_distances = []
#creating dictionaries wot store the distance as the key and the cor-ordinate as the value
eu_distance_point_dict = {}
man_distance_point_dict = {}
min_distance_point_dict = {}
cheby_distance_point_dict = {}
cosine_distance_point_dict = {}

# this loop will iterate through the whole column of displacement and weight and create 
# data points for each record, and then calculate the distances and store them in necessary data structure
for i in range(len(data.displacement)):
    s.append(data.iloc[i,2])
    s.append(data.iloc[i,4])
    #Euclidean Distance
    eu_dist = distance.euclidean(p,s)
    eu_distance_point_dict[eu_dist] = [s[0], s[1]]
    #Manhattan Distance
    manhattan_dist = scipy.spatial.distance.cityblock(p,s)
    man_distance_point_dict[manhattan_dist] = [s[0], s[1]]
    #Minkowski Distance
    min_dist = scipy.spatial.distance.minkowski(p,s,7)
    min_distance_point_dict[min_dist] = [s[0], s[1]]
    #Chebyshev Distance
    cheby_dist = scipy.spatial.distance.chebyshev(p,s)
    cheby_distance_point_dict[cheby_dist] = [s[0], s[1]]
    #Cosine Distance
    cosine_dist = scipy.spatial.distance.cosine(p,s)
    cosine_distance_point_dict[cosine_dist] = [s[0], s[1]]
    s.clear()                          #clearing s so that the next set of co-ordinates can be generated
    #print("the euclidean distance for", i,"point is:",eu_dist)
    #print("the manhanttan distance for", i,"point is :", manhattan_dist)
    #print("the minkowski distance for", i,"point is :", min_dist)
    #print("the Chebyshev distance for", i,"point is :", cheby_dist)
    #print("the cosine distance for", i,"point is :", cosine_dist)
    #Append the various distances in the dictionary for plotting the graphs
    eu_distances.append(eu_dist)
    man_distances.append(manhattan_dist)
    min_distances.append(min_dist)
    cheby_distances.append(cheby_dist)
    cosine_distances.append(cosine_dist)
#print(eu_distances)   
#print(man_distances)
#print(min_distances)
#print(cheby_distances)
#print(cosine_distances)

# 6 closest distances for each 
eu_distances.sort()
print("6 closest points to P using euclidean distance :",(eu_distances[0:6]))

man_distances.sort()
print("\n6 closest points to P using manhattan distance :",(man_distances[0:6]))

min_distances.sort()
print("\n6 closest points to P using minkowski distance :",(min_distances[0:6]))

cheby_distances.sort()
print("\n6 closest points to P using chebyshev distance :",(cheby_distances[0:6]))

cosine_distances.sort()
print("\n6 closest points to P using cosine distance :",(cosine_distances[0:6]))

# d. i. Create plots, one for each distance measure. Place P on the plot
# and mark the 20 closest points. To mark them, you could use different colors
# or shapes. Make sure the points can be uniquely identifed. 

import matplotlib.pyplot as plt

# Creates a sorted dictionary (sorted by key) for each distance type
from collections import OrderedDict

eu_dict_sorted = OrderedDict(sorted(eu_distance_point_dict.items()))

man_dict_sorted = OrderedDict(sorted(man_distance_point_dict.items()))

min_dict_sorted = OrderedDict(sorted(min_distance_point_dict.items()))

cheby_dict_sorted = OrderedDict(sorted(cheby_distance_point_dict.items()))

cosine_dict_sorted = OrderedDict(sorted(cosine_distance_point_dict.items()))

# creating lists to separatley store x and y co-ordinates , making it easier to plot
x_coordinates = []
y_coordinates = []
# Storing the distances by referencing key value from distances dictionary 
eu_distance_vec = []
man_distance_vec = []
min_distance_vec = []
cheby_distance_vec = []
cosine_distance_vec = []

colors = ['#000000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000']

x_coordinates.append(p[0])
y_coordinates.append(p[1])

eu_distance_vec.append(0)
man_distance_vec.append(0)
min_distance_vec.append(0)
cheby_distance_vec.append(0)
cosine_distance_vec.append(0)
# sorting and getting the co-ordinates required for plotting for each distance
for z in eu_dict_sorted:   # z is key
  x_coordinates.append(eu_dict_sorted[z][0])
  y_coordinates.append(eu_dict_sorted[z][1])
  eu_distance_vec.append(z)

for z in man_dict_sorted:   # z is key
  x_coordinates.append(man_dict_sorted[z][0])
  y_coordinates.append(man_dict_sorted[z][1])
  man_distance_vec.append(z)

for z in min_dict_sorted:   # z is key
  x_coordinates.append(min_dict_sorted[z][0])
  y_coordinates.append(min_dict_sorted[z][1])
  min_distance_vec.append(z)

for z in cheby_dict_sorted:   # z is key
  x_coordinates.append(cheby_dict_sorted[z][0])
  y_coordinates.append(cheby_dict_sorted[z][1])
  cheby_distance_vec.append(z)

for z in cosine_dict_sorted:   # z is key
  x_coordinates.append(cosine_dict_sorted[z][0])
  y_coordinates.append(cosine_dict_sorted[z][1])
  cosine_distance_vec.append(z)

#Scatterplot of Displacement vs Weight for Euclidean Distance
plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

for i in range(21):
    plt.scatter(x_coordinates[i], y_coordinates[i], 21, label="distance='{0}'".format(eu_distance_vec[i]), color= colors[i])

plt.legend(numpoints=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Scatterplot of Displacement vs Weight using Euclidean Distance", fontsize=22)
plt.xlabel("Displacement")
plt.ylabel("Weight")

#Scatterplot of Displacement vs Weight for Manhattan Distance
plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

for i in range(21):
    plt.scatter(x_coordinates[i], y_coordinates[i], 21, label="distance='{0}'".format(man_distance_vec[i]), color= colors[i])
    plt.legend(numpoints=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Scatterplot of Displacement vs Weight using Manhattan Distance", fontsize=22)
plt.xlabel("Displacement")
plt.ylabel("Weight")

#Scatterplot of Displacement vs Weight for Min Distance
plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

for i in range(21):
    plt.scatter(x_coordinates[i], y_coordinates[i], 21, label="distance='{0}'".format(min_distance_vec[i]), color= colors[i])
    plt.legend(numpoints=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Scatterplot of Displacement vs Weight using Minkowski Distance", fontsize=22)
plt.xlabel("Displacement")
plt.ylabel("Weight")

#Scatterplot of Displacement vs Weight for Chebychev Distance
plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

for i in range(21):
    plt.scatter(x_coordinates[i], y_coordinates[i], 21, label="distance='{0}'".format(cheby_distance_vec[i]), color= colors[i])
    plt.legend(numpoints=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Scatterplot of Displacement vs Weight using Chebychev Distance", fontsize=22)
plt.xlabel("Displacement")
plt.ylabel("Weight")

#Scatterplot of Displacement vs Weight for Chebychev Distance
plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

for i in range(21):
    plt.scatter(x_coordinates[i], y_coordinates[i], 21, label="distance='{0}'".format(cosine_distance_vec[i]), color= colors[i])
    plt.legend(numpoints=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Scatterplot of Displacement vs Weight using Cosine Distance", fontsize=22)
plt.xlabel("Displacement")
plt.ylabel("Weight")

plt.show()
# d. ii. Verify if the set of points is the same across all the distance measures.
#If there is any big difference, briefly explain why it is.
print("The graphs indicate that the location of all points are same with respect to the point P. \nThe only difference is the value of the shortest distance of each point from point P")