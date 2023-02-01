import random

import numpy as np


def train_test_split(data, test_size: float):
    """
    Splits input data into training and testing sets of specified size.

    :param data: input data (Pandas DataFrame).
    :param test_size: size for test set relative to entire dataset (0 to 1).
    :return: train_df and test_df.
    """
    copy_df = data.copy(deep=True)  # Create deep copy of input data.
    indices = copy_df.index.tolist()  # Get indices from data.
    # Calculate k based on amount of rows in data, then cast to int:
    k = int(np.round((copy_df.shape[0] * test_size), 0))
    selected_indices = random.sample(population=indices, k=k)  # Select random indices where n=test_size.

    test_df = copy_df.loc[selected_indices]
    train_df = copy_df.drop(selected_indices)

    return train_df, test_df
