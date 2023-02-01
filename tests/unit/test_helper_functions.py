from unittest import TestCase

import pandas as pd

from src.decision_tree import check_purity
from src.helper_functions import train_test_split


class HelperFunctionsTestSuite(TestCase):
    def setUp(self) -> None:
        self.mock_data = pd.DataFrame({
            'person': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'sex': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
        })

    def test_train_test_split(self):
        train_df, test_df = train_test_split(self.mock_data, 0.2)

        self.assertEqual(8, len(train_df))
        self.assertEqual(2, len(test_df))

    def test_check_purity(self):
        is_pure = check_purity(labels=self.mock_data['sex'])

        self.assertFalse(is_pure)

        pure_data = self.mock_data[self.mock_data.sex == 'Male']  # Fetch pure dataset.
        is_pure = check_purity(labels=pure_data['sex'])

        self.assertTrue(is_pure)
