import numpy as np


def check_purity(labels) -> bool:
    """
    Evaluates if the input data is pure.

    :param labels: labels for the provided input data (2D Numpy array).
    :return: boolean for purity of the data.
    """
    unique_classes = np.unique(labels)

    # If there is one class the data is pure, so return True:
    if len(unique_classes) == 1:
        return True

    return False


def classify_data(labels) -> str:
    """
    Classify the given input using majority voting (simply select most common label).

    :param labels: labels for the provided input data (2D Numpy array).
    :return: string with the class.
    """
    unique_classes, class_count = np.unique(labels, return_counts=True)
    index = class_count.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data):
    """
    Find the potential splits in the data. This data should not contain the label and must be all numbers.

    :param data: input data (2D Numpy array).
    :return: dictionary with all potential splits.
    """
    potential_splits = {}
    n_columns = data.shape[1]

    for i in range(n_columns):
        potential_splits[i] = []  # Create list for column.
        values = data[:, i]  # Fetch all rows from specific column i.
        unique_values = np.unique(values)

        for j in range(len(unique_values)):
            if j > 0:
                current_value = unique_values[j]
                previous_value = unique_values[j - 1]
                potential_split = (current_value + previous_value) / 2
                potential_splits[i].append(potential_split)

    return potential_splits


def split_data(data, split_column: int, split_threshold: float):
    """
    Split the data along a given column with a given split threshold.

    :param data: input data to be split.
    :param split_column: column index along which to split the data.
    :param split_threshold: value that determines at which point to split the data.
    :return: arrays with data below threshold and with data above threshold.
    """
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_threshold]
    data_above = data[split_column_values > split_threshold]

    return data_below, data_above


def calculate_entropy(labels) -> float:
    """
    Calculate the entropy (randomness) for a given dataset. Lower variance means lower entropy, higher variance
    means higher entropy.

    :param labels: labels for input data.
    :return: entropy value (float between 0 and 1).
    """
    counts = np.unique(labels, return_counts=True)[1]  # Get count for each class in labels.
    probabilities = counts / counts.sum()  # Calculate probabilities for each class.

    return float(np.sum(probabilities * - np.log2(probabilities)))


def calculate_overall_entropy(data_below, data_above) -> float:
    """
    Calculates the overall entropy (variance) for the entire provided dataset.

    :param data_below: data below the split threshold.
    :param data_above: data above the split threshold.
    :return: the overall entropy value (float between 0 and 1).
    """

    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    return p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above)


def determine_best_split(data):
    """
    Find the best split column and threshold (split value) for the provided data.

    :param data: input data.
    :return: column and threshold (split value) for best split.
    """
    best_split_column = None
    best_split_value = None
    overall_entropy = np.inf
    potential_splits = get_potential_splits(data)  # Get the potential splits for the provided data.

    for i in potential_splits:
        for value in potential_splits[i]:
            data_below, data_above = split_data(data=data, split_column=i, split_threshold=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy < overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = i
                best_split_value = value

    return best_split_column, best_split_value


class DecisionTree:
    """
    A decision tree model that used 2D Numpy arrays as input data. The data must contain a label.
    Based on the guide by Sebastian Mantey.
    """

    def __init__(self, min_samples_split=5, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _run_algorithm(self, x, y, counter=0):
        """
        Recursive function that runs the decision tree algorithm.

        :param x: input data (2D Numpy array).
        :param y: labes for the input data (2D Numpy array).
        :param counter: value to save ...
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
            potential_splits = get_potential_splits(x)
            split_col, split_value = determine_best_split(x)
            # Split the data:
            data_below, data_above = split_data(data, split_col, split_value)

            # Instantiate sub-tree:
            question = f"Column {split_col} <= {split_value}"
            sub_tree = {question: []}  # Append yes/no answers here.

            # Then split into data and labels again:
            x_below = data_below[:, :-1]
            x_above = data_above[:, :-1]
            y_below = data_below[:, -1]
            y_above = data_above[:, -1]

            # Find answer to question (recursive):
            yes_answer = self._run_algorithm(x_below, y_below, counter)
            no_answer = self._run_algorithm(x_above, y_above, counter)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree