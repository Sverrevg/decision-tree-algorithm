from unittest import TestCase

import pandas as pd

from src.impurityOOOOLD import calculate_gini_impurity, calculate_entropy


class ImpurityTestSuite(TestCase):
    def setUp(self) -> None:
        self.mock_data = pd.DataFrame({
            'person': [1, 2, 3, 4],
            'sex': ['Female', 'Male', 'Female', 'Male']
        })

    def test_gini_index(self):
        expected = 0.5  # Expect impurity to be max (0.5) as the distribution is exactly 50% in the mock data.
        impurity = calculate_gini_impurity(self.mock_data['sex'])

        self.assertEqual(expected, impurity)

    def test_entropy(self):
        expected = 0.9999
        entropy = calculate_entropy(self.mock_data['sex'])

        self.assertAlmostEqual(expected, entropy, 3)
