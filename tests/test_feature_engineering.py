import unittest
import pandas as pd
from your_module import (
    create_aggregate_features,
    encode_categorical_variables,
    normalize_data,
    handle_missing_values
)  # Adjust import based on your project structure

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'Amount': [15000, 20000, 25000],
            'Value': [50000, 60000, 70000],
            'PricingStrategy': ['A', 'B', 'A'],
            'MissingCol': [1, None, 3]
        })
        self.expected_aggregated_df = pd.DataFrame({
            'Total_Amount': [60000],
            'Average_Amount': [20000],
            'Transaction_Count': [3],
            'Std_Amount': [5000]  # Adjust based on expected calculation
        })

    def test_create_aggregate_features(self):
        """Test the aggregate features creation."""
        aggregated_df = create_aggregate_features(self.df)
        pd.testing.assert_frame_equal(
            aggregated_df.reset_index(drop=True),
            self.expected_aggregated_df.reset_index(drop=True),
            check_dtype=False
        )

    def test_encode_categorical_variables(self):
        """Test categorical variable encoding."""
        encoded_df = encode_categorical_variables(self.df)
        expected_encoded_columns = ['Amount', 'Value', 'PricingStrategy_A', 'PricingStrategy_B', 'MissingCol']
        for col in expected_encoded_columns:
            self.assertIn(col, encoded_df.columns)

    def test_handle_missing_values(self):
        """Test missing values handling."""
        df_with_nan = self.df.copy()
        df_with_nan['MissingCol'] = df_with_nan['MissingCol'].fillna(0)
        filled_df = handle_missing_values(df_with_nan)
        self.assertFalse(filled_df['MissingCol'].isnull().any(), "Missing values should be handled")

    def test_normalize_data(self):
        """Test normalization of numerical features."""
        normalized_df = normalize_data(self.df[['Amount', 'Value']])
        self.assertAlmostEqual(normalized_df['Amount'].min(), 0, places=2)
        self.assertAlmostEqual(normalized_df['Amount'].max(), 1, places=2)
        self.assertAlmostEqual(normalized_df['Value'].min(), 0, places=2)
        self.assertAlmostEqual(normalized_df['Value'].max(), 1, places=2)

    def tearDown(self):
        # Clean up after each test if necessary
        pass

if __name__ == '__main__':
    unittest.main()
