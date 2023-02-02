from pprint import pprint

import numpy as np

from src.algorithm_functions import split_data, check_purity, classify_data, determine_best_split


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

    def predict(self, x):
        """
        Classifies the given sample(s).

        :param x: input data (2D Numpy array).
        :return: an array with all answers.
        """
        answers = []

        # Ensure input data is in the correct shape (1, n):
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])

        for i in range(0, x.shape[0]):
            answers.append(self._classify(x[i], self.tree))

        return answers

    def fit(self, x, y):
        """
        Wrapper for the algorithm function. Saves the output tree to a local dictionary for later use.

        :param x: input data (2D Numpy array).
        :param y: labels for the input data (2D Numpy array).
        """
        self.tree = self._fit_algorithm(x, y)

    def _classify(self, x, tree) -> int:
        """
        Recursive algorithm used to classify the given sample.

        :param x: input data.
        :param tree: decision tree that will be used to make the classification.
        :return: class (int).
        """
        if not tree:
            raise Exception('Please fit the model before classification.')

        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split()

        # Ask question:
        if x[int(feature_name)] <= float(value):
            answer = tree[question]['yes']
        else:
            answer = tree[question]['no']

        # Base case:
        if not isinstance(answer, dict):
            return int(answer)
        # Recursive part:
        else:
            return self._classify(x, answer)

    def _fit_algorithm(self, x, y, counter=0):
        """
        Recursive function that runs the decision tree algorithm.

        :param x: input data (2D Numpy array).
        :param y: labels for the input data (2D Numpy array).
        :param counter: value to save amount of times the function was called.
        :return:
        """
        # First join x and y to a new array for easy splitting:
        if len(y.shape) == 1:  # Ensure y has shape of (n, 1).
            y = y.reshape(y.shape[0], 1)
        data = np.append(x, y, axis=1)

        # Base case:
        if check_purity(y) or len(data) < self.min_samples_split or counter == self.max_depth:
            return classify_data(y)
        # Recursive part of the algorithm:
        else:
            counter += 1
            # Get splits:
            split_col, split_value = determine_best_split(x, self.criterion)
            # Split the data:
            data_below, data_above = split_data(data, split_col, split_value)

            # Instantiate sub-tree:
            question = f"{split_col} <= {split_value}"
            sub_tree = {question: {}}  # Stores yes/no answers.

            # Then split into data and labels again:
            x_below = data_below[:, :-1]
            x_above = data_above[:, :-1]
            y_below = data_below[:, -1]
            y_above = data_above[:, -1]

            # Find answer to question (recursive):
            yes_answer = self._fit_algorithm(x_below, y_below, counter)
            no_answer = self._fit_algorithm(x_above, y_above, counter)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question]['yes'] = yes_answer
                sub_tree[question]['no'] = no_answer

            return sub_tree

    def accuracy(self, x, y):
        """
        Test the accuracy of the current decision tree.

        :param x: input data to test with (2D Numpy array).
        :param y: labels for the input data (2D Numpy array).
        :return: accuracy score.
        """
        if not self.tree:
            raise Exception('Please fit the model before calculating accuracy.')

        predicted = np.array(self.predict(x))
        # Create list of predictions that don't match the label:
        diffs = list(np.array(y - predicted).nonzero()[0])

        # Calculate accuracy by dividing amount wrong by total amount:
        return 1 - (len(diffs) / len(y))

    def print_tree(self):
        """
        Prints the structure of the decision tree using pretty print.
        """
        if not self.tree:
            raise Exception('No tree found to print.')

        pprint(self.tree)
