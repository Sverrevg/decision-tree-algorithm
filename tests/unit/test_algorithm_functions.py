from unittest import TestCase

import numpy as np
import pandas as pd

from src.algorithm_functions import check_purity, get_potential_splits, split_data
from src.decision_tree import classify_data, determine_best_split


class AlgorithmTestSuite(TestCase):
    def setUp(self) -> None:
        np.random.seed(123)  # Set seed.
        self.mock_data = pd.DataFrame({
            'person': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'age': [45, 23, 65, 34, 75, 23, 46, 82, 26, 32],
            'sex': [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        })

    def test_check_purity(self):
        is_pure = check_purity(labels=self.mock_data['sex'])

        self.assertFalse(is_pure)

        pure_data = self.mock_data[self.mock_data.sex == 0]  # Fetch pure dataset.
        is_pure = check_purity(labels=pure_data['sex'])

        self.assertTrue(is_pure)

    def test_classify_data(self):
        classification = classify_data(labels=self.mock_data['sex'])

        self.assertEqual(0, classification)

        pure_data = self.mock_data[self.mock_data.sex == 1]  # Fetch pure dataset.
        classification = classify_data(labels=pure_data['sex'])

        self.assertEqual(1, classification)

    def test_get_potential_splits(self):
        expected = {0: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                    1: [24.5, 29.0, 33.0, 39.5, 45.5, 55.5, 70.0, 78.5]}
        potential_splits = get_potential_splits(x_data=self.mock_data[['person', 'age']].values)

        self.assertEqual(len(expected), len(potential_splits))

    def test_split_data(self):
        data_below, data_above = split_data(data=self.mock_data[['person', 'age']].values, split_column=1,
                                            split_threshold=50)

        self.assertEqual(7, len(data_below))
        self.assertEqual(3, len(data_above))

    def test_determine_best_split(self):
        best_split = determine_best_split(x_data=self.mock_data[['person', 'age']].values, criterion='entropy')

        self.assertEqual((1, 39.5), best_split)
