import unittest
import pandas as pd
from your_module import load_data  # Adjust import based on your project structure

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        # This method will run before each test
        self.valid_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\data.csv'  # Valid data file path
        self.valid_definitions_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\Xente_Variable_Definitions.csv'  # Variable definitions path
        self.invalid_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\invalid_data.csv'  # Invalid file path

    def test_load_data_valid(self):
        """Test if load_data function works with a valid file path."""
        data = load_data(self.valid_file_path)
        self.assertIsInstance(data, pd.DataFrame, "The loaded data should be a pandas DataFrame")
        self.assertFalse(data.empty, "The loaded DataFrame should not be empty")

    def test_load_definitions_valid(self):
        """Test if load_data function works with the variable definitions file."""
        definitions = load_data(self.valid_definitions_path)
        self.assertIsInstance(definitions, pd.DataFrame, "The loaded definitions should be a pandas DataFrame")
        self.assertFalse(definitions.empty, "The loaded definitions DataFrame should not be empty")

    def test_load_data_invalid(self):
        """Test if load_data function raises an error with an invalid file path."""
        with self.assertRaises(FileNotFoundError):
            load_data(self.invalid_file_path)

    def test_load_data_shape(self):
        """Test if the loaded DataFrame has the expected shape."""
        data = load_data(self.valid_file_path)
        expected_shape = (100, 5)  # Adjust based on your expected shape
        self.assertEqual(data.shape, expected_shape, f"Expected shape {expected_shape}, but got {data.shape}")

    def tearDown(self):
        # Clean up after each test if necessary
        pass

if __name__ == '__main__':
    unittest.main()
