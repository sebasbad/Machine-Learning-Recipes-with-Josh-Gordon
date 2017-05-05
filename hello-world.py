# Hello World - Machine Learning Recipes #1
# https://www.youtube.com/watch?v=cKxRvEZd3Mw

#import sklearn
from sklearn import tree

# Supervised Learning Recipe

# 1.Collect training data
# scikit-learn user real-valuled features
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#labels = ["apple", "apple", "orange", "orange"]
labels = [0, 0, 1, 1]

# 2.Train classifier

# Decision tree
clf = tree.DecisionTreeClassifier()
# fit ~ find patterns in data
clf.fit(features, labels)

# 3.Make predictions
print clf.predict([[160, 0]])
