from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.decision_tree import DecisionTree

wine = datasets.load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

decision_tree = DecisionTree(min_samples_split=5, max_depth=6)
decision_tree.fit(X_train, y_train)
# decision_tree.print_tree()
# print(f"Predicted class(es): {decision_tree.predict(X)}")
print(f"Accuracy: {decision_tree.accuracy(X_test, y_test)}")
