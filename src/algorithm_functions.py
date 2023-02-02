import numpy as np

from src.criterion_functions import entropy, gini


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


def get_potential_splits(x):
    """
    Find the potential splits in the data. This data should not contain the label and must be all numbers.

    :param x: input data (2D Numpy array).
    :return: dictionary with all potential splits.
    """
    potential_splits = {}
    n_columns = x.shape[1]

    for i in range(n_columns):
        potential_splits[i] = []  # Create list for column.
        values = x[:, i]  # Fetch all rows from specific column i.
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


def determine_best_split(x, criterion, round=2):
    """
    Find the best split column and threshold (split value) for the provided data.

    :param x: input data.
    :param criterion: criterion method to use.
    :param round: how many decimals to keep after rounding.
    :return: column and threshold (split value) for best split.
    """
    best_split_column = None
    best_split_value = None
    overall_impurity = np.inf
    potential_splits = get_potential_splits(x)  # Get the potential splits for the provided data.

    for i in potential_splits:
        for value in potential_splits[i]:
            if criterion == 'entropy':
                data_below, data_above = split_data(data=x, split_column=i, split_threshold=value)
                current_overall_impurity = entropy(data_below, data_above)

            elif criterion == 'gini':
                current_overall_impurity = gini(x)
            else:
                raise Exception("'Please provide a valid criterion ('entropy', 'gini')'")

            if current_overall_impurity < overall_impurity:
                overall_impurity = current_overall_impurity
                best_split_column = i
                best_split_value = value

    return best_split_column, np.round(best_split_value, round)
