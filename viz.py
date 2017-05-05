# Visualizing a Decision Tree - Machine Learning Recipes #2
# https://www.youtube.com/watch?v=tNa99PG8hR8

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Import dataset
iris = load_iris()

#print iris.feature_names
#print iris.target_names
#print iris.data[0]
#print iris.target[0]

# Iterate and print the entire dataset
#for i in range(len(iris.target)):
#    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

# Split the data: training set and test set

# test data set will be the first sample for each type of iris flower
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train a classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# predict labels for new flowers
print test_target
print clf.predict(test_data)

# dependencies to install: pydot, graphviz
# https://anaconda.org/anaconda/pydot
# http://www.graphviz.org/Download_macos.php

# viz code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file = dot_data,
                        feature_names = iris.feature_names,
                        class_names = iris.target_names,
                        filled = True,
                        rounded = True,
                        impurity = False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print test_data[0], test_target[0]
print test_data[1], test_target[1]
print test_data[2], test_target[2]

print iris.feature_names, iris.target_names
