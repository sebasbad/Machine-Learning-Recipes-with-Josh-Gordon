# Let's Write a Pipeline - Machine Learning Recipes #4

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)
my_classifier2.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print predictions
predictions2 = my_classifier2.predict(X_test)
print predictions2

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
print accuracy_score(y_test, predictions2)
