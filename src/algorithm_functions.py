from typing import Any

import numpy as np

from src.criterion_functions import entropy, gini_coefficient


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


def classify_data(labels) -> Any:
    """
    Classify the given input using majority voting (simply select most common label).

    :param labels: labels for the provided input data (2D Numpy array).
    :return: string with the class.
    """
    unique_classes, class_count = np.unique(labels, return_counts=True)
    index = class_count.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(x_data):
    """
    Find the potential splits in the data. This data should not contain the label and must be all numbers.

    :param x_data: input data (2D Numpy array).
    :return: dictionary with all potential splits.
    """
    potential_splits = {}
    n_columns = x_data.shape[1]

    for i in range(n_columns):
        potential_splits[i] = []  # Create list for column.
        values = x_data[:, i]  # Fetch all rows from specific column i.
        unique_values = np.unique(values)

        for j, value in enumerate(unique_values):
            if j > 0:
                previous_value = unique_values[j - 1]
                potential_split = (value + previous_value) / 2
                potential_splits[i].append(potential_split)

    return potential_splits


def split_x_y_data(x_data, y_data, split_column: int, split_threshold: float):
    """
    Split the data along a given column with a given split threshold.

    :param x_data: input data to be split.
    :param y_data: labels for input data.
    :param split_column: column index along which to split the data.
    :param split_threshold: value that determines at which point to split the data.
    :return: arrays with data below threshold and with data above threshold.
    """
    # First join x and y to a new array for easy splitting:
    if len(y_data.shape) == 1:  # Ensure y has shape of (n, 1).
        y_data = y_data.reshape(y_data.shape[0], 1)
    data = np.append(x_data, y_data, axis=1)

    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_threshold]
    data_above = data[split_column_values > split_threshold]

    # Then separate into data and labels again:
    x_below = data_below[:, :-1]
    x_above = data_above[:, :-1]
    y_below = data_below[:, -1]
    y_above = data_above[:, -1]

    return x_below, x_above, y_below, y_above


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


def determine_best_split(x_data, criterion, rounding_factor=2):
    """
    Find the best split column and threshold (split value) for the provided data.

    :param x_data: input data.
    :param criterion: criterion method to use.
    :param rounding_factor: how many decimals to keep after rounding.
    :return: column and threshold (split value) for best split.
    """
    best_split_column = None
    best_split_value = None
    overall_impurity = np.inf
    potential_splits = get_potential_splits(x_data)  # Get the potential splits for the provided data.

    # Loop through each column:
    for column, values in potential_splits.items():
        # Loop through values in column:
        for value in values:
            x_below, x_above = split_data(data=x_data, split_column=column, split_threshold=value)
            if criterion == 'entropy':
                current_overall_impurity = entropy(x_below, x_above)

            elif criterion == 'gini':
                current_overall_impurity = gini_coefficient(x_below, x_above)
            else:
                raise ValueError("'Please provide a valid criterion ('entropy', 'gini')'")

            # Select column and value with the smallest impurity
            if current_overall_impurity < overall_impurity:
                overall_impurity = current_overall_impurity
                best_split_column = column
                best_split_value = value

    return best_split_column, np.round(best_split_value, rounding_factor)
