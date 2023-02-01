from pprint import pprint

from sklearn import datasets

from src.decision_tree import DecisionTree

# Import some data to play with:
iris = datasets.load_iris()
X = iris.data
y = iris.target

decision_tree = DecisionTree(min_samples_split=20)
pprint(decision_tree._run_algorithm(X, y))
