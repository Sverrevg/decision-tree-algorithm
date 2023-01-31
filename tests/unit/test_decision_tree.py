from unittest import TestCase

import pandas as pd

from src.decision_tree import classify_data, get_potential_splits, split_data, calculate_entropy, \
    calculate_overall_entropy, determine_best_split


class DecisionTreeTestSuite(TestCase):
    def setUp(self) -> None:
        self.mock_data = pd.DataFrame({
            'person': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'age': [45, 23, 65, 34, 75, 23, 46, 82, 26, 32],
            'sex': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
        })

    def test_classify_data(self):
        classification = classify_data(labels=self.mock_data['sex'])

        self.assertEqual('Female', classification)

        pure_data = self.mock_data[self.mock_data.sex == 'Male']  # Fetch pure dataset.
        classification = classify_data(labels=pure_data['sex'])

        self.assertEqual('Male', classification)

    def test_get_potential_splits(self):
        expected = {0: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                    1: [24.5, 29.0, 33.0, 39.5, 45.5, 55.5, 70.0, 78.5]}
        potential_splits = get_potential_splits(data=self.mock_data[['person', 'age']].values)

        self.assertEqual(expected, potential_splits)

    def test_split_data(self):
        data_below, data_above = split_data(data=self.mock_data[['person', 'age']].values, split_column=1,
                                            split_threshold=50)

        self.assertEqual(7, len(data_below))
        self.assertEqual(3, len(data_above))

    def test_calculate_entropy(self):
        data_below = [1, 0, 1, 0, 0, 0, 1]
        data_above = [0, 0, 0]

        entropy_below = calculate_entropy(data_below)
        entropy_above = calculate_entropy(data_above)

        self.assertAlmostEqual(0.985, entropy_below, 3)
        self.assertEqual(0., entropy_above)

    def test_calculate_overall_entropy(self):
        data_below = [1, 0, 1, 0, 0, 0, 1]
        data_above = [0, 0, 0]

        overall_entropy = calculate_overall_entropy(data_below=data_below, data_above=data_above)

        self.assertAlmostEqual(0.690, overall_entropy, 3)

    def test_determine_best_split(self):
        best_split = determine_best_split(self.mock_data[['person', 'age']].values)

        self.assertEqual((1, 39.5), best_split)
