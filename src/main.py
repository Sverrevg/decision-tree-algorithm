def train_tree(data, y, target_factor, max_depth=None, min_samples_split=None, min_information_gain=1e-20, counter=0,
               max_categories=20):
    """
    Train a decision tree model.

    :param data:
    :param y:
    :param target_factor:
    :param max_depth:
    :param min_samples_split:
    :param min_information_gain:
    :param counter:
    :param max_categories:
    :return:
    """
