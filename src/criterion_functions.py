import numpy as np


def _calculate_entropy(labels) -> float:
    """
    Calculate the entropy (randomness) for a given dataset. Lower variance means lower entropy, higher variance
    means higher entropy.

    :param labels: labels for input data.
    :return: entropy value (float between 0 and 1).
    """
    counts = np.unique(labels, return_counts=True)[1]  # Get count for each class in labels.
    probabilities = counts / counts.sum()  # Calculate probabilities for each class.

    return float(np.sum(probabilities * - np.log2(probabilities)))


def entropy(data_below, data_above) -> float:
    """
    Calculates the overall entropy (variance) for the entire provided dataset.

    :param data_below: data below the split threshold.
    :param data_above: data above the split threshold.
    :return: the overall entropy value (float between 0 and 1).
    """

    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    return p_data_below * _calculate_entropy(data_below) + p_data_above * _calculate_entropy(data_above)


def gini_coefficient(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / y.shape[0]
    sq_sum = np.sum(probabilities ** 2)

    return 1 - sq_sum
