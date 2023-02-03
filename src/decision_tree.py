from pprint import pprint

import numpy as np

from src.algorithm_functions import split_x_y_data, check_purity, classify_data, determine_best_split


class DecisionTree:
    """
    A decision tree model that used 2D Numpy arrays as input data. The data must contain a label.
    Based on the guide by Sebastian Mantey.
    """

    def __init__(self, criterion='entropy', min_samples_split=5, max_depth=5):
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = {}

    def predict(self, x_data):
        """
        Classifies the given sample(s).

        :param x_data: input data (2D Numpy array).
        :return: an array with all answers.
        """
        answers = []

        # Ensure input data is in the correct shape (1, n):
        if len(x_data.shape) == 1:
            x_data = x_data.reshape(1, x_data.shape[0])

        for i in range(0, x_data.shape[0]):
            answers.append(self._classify(x_data[i], self.tree))

        return answers

    def fit(self, x_data, y_data):
        """
        Wrapper for the algorithm function. Saves the output tree to a local dictionary for later use.

        :param x_data: input data (2D Numpy array).
        :param y_data: labels for the input data (2D Numpy array).
        """
        self.tree = self._fit_algorithm(x_data, y_data)

    def _classify(self, x_data, tree) -> int:
        """
        Recursive algorithm used to classify the given sample.

        :param x_data: input data.
        :param tree: decision tree that will be used to make the classification.
        :return: class (int).
        """
        if not tree:
            raise RuntimeError('Please fit the model before classification.')

        question = list(tree.keys())[0]
        feature_name, _, value = question.split()

        # Ask question:
        if x_data[int(feature_name)] <= float(value):
            answer = tree[question]['yes']
        else:
            answer = tree[question]['no']

        # Base case:
        if not isinstance(answer, dict):
            return int(answer)

        # Recursive part:
        return self._classify(x_data, answer)

    def _fit_algorithm(self, x_data, y_data, counter=0):
        """
        Recursive function that runs the decision tree algorithm.

        :param x_data: input data (2D Numpy array).
        :param y_data: labels for the input data (2D Numpy array).
        :param counter: value to save amount of times the function was called.
        :return:
        """
        # Base case:
        if check_purity(y_data) or len(y_data) < self.min_samples_split or counter == self.max_depth:
            return classify_data(y_data)

        # Recursive part of the algorithm:
        counter += 1
        # Get splits:
        split_col, split_value = determine_best_split(x_data, self.criterion)
        # Split the data:
        x_below, x_above, y_below, y_above = split_x_y_data(x_data, y_data, split_col, split_value)

        # Instantiate sub-tree:
        question = f"{split_col} <= {split_value}"
        sub_tree = {question: {}}  # Stores yes/no answers.

        # Find answer to question (recursive):
        yes_answer = self._fit_algorithm(x_below, y_below, counter)
        no_answer = self._fit_algorithm(x_above, y_above, counter)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question]['yes'] = yes_answer
            sub_tree[question]['no'] = no_answer

        return sub_tree

    def accuracy(self, x_data, y_data):
        """
        Test the accuracy of the current decision tree.

        :param x_data: input data to test with (2D Numpy array).
        :param y_data: labels for the input data (2D Numpy array).
        :return: accuracy score.
        """
        if not self.tree:
            raise RuntimeError('Please fit the model before calculating accuracy.')

        predicted = np.array(self.predict(x_data))
        # Create list of predictions that don't match the label:
        diffs = list(np.array(y_data - predicted).nonzero()[0])

        # Calculate accuracy by dividing amount wrong by total amount:
        return 1 - (len(diffs) / len(y_data))

    def print_tree(self) -> None:
        """
        Prints the structure of the decision tree using pretty print.
        """
        if not self.tree:
            raise RuntimeError('No tree found to print.')

        pprint(self.tree)
