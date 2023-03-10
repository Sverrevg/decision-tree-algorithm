from unittest import TestCase

import numpy as np
import pandas as pd

from src.criterion_functions import _calculate_entropy, entropy, gini_coefficient


class CriterionFunctionsTestSuite(TestCase):
    def setUp(self) -> None:
        np.random.seed(123)  # Set seed.
        self.mock_data = pd.DataFrame({
            'person': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'sex': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
        })

    def test_calculate_entropy(self):
        data_below = [1, 0, 1, 0, 0, 0, 1]
        data_above = [0, 0, 0]

        entropy_below = _calculate_entropy(data_below)
        entropy_above = _calculate_entropy(data_above)

        self.assertAlmostEqual(0.985, entropy_below, 3)
        self.assertEqual(0., entropy_above)

    def test_calculate_overall_entropy(self):
        data_below = [1, 0, 1, 0, 0, 0, 1]
        data_above = [0, 0, 0]

        overall_entropy = entropy(data_below=data_below, data_above=data_above)

        self.assertAlmostEqual(0.690, overall_entropy, 3)

    def test_calculate_gini(self):
        x = np.random.rand(500)
        y = np.random.rand(500)
        gini = gini_coefficient(x, y)

        self.assertAlmostEqual(0.33, gini, 2)
