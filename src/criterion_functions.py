import numpy as np


def _calculate_entropy(labels) -> float:
    """
    Calculate the entropy (randomness) for a given label distribution. Lower variance means lower entropy, higher
    variance means higher entropy.

    :param labels: labels for input data.
    :return: entropy value (float between 0 and 1).
    """
    counts = np.unique(labels, return_counts=True)[1]  # Get count for each class in labels.
    probabilities = counts / counts.sum()  # Calculate probabilities for each class.

    return float(np.sum(probabilities * - np.log2(probabilities)))


def _calc_gini_coefficient(input_data):
    """
    Calculate gini coefficient using numpy vectorization.
    Source: https://www.statology.org/gini-coefficient-python/

    :param input_data: input data (2D Numpy array).
    :return: Gini coefficient (float).
    """
    total = 0
    for i, value, in enumerate(input_data[:-1], 1):
        total += np.sum(np.abs(value - input_data[i:]))

    return total / (len(input_data) ** 2 * np.mean(input_data))


def entropy(data_below: [], data_above: []) -> float:
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


def gini_coefficient(data_below: [], data_above: []):
    """
    Calculates the overall gini coefficient for the entire provided dataset.
    :param data_below:
    :param data_above:
    :return:
    """
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    return p_data_below * _calc_gini_coefficient(data_below) + p_data_above * _calc_gini_coefficient(data_above)
