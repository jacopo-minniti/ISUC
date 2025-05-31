"""
Unit tests for the base classes and interfaces.
"""
import unittest
import torch
import tempfile
import json
import os
from src.base import ActivationsDataset, BaseUncertaintyClassifier, BaseDatasetProcessor


class TestActivationsDataset(unittest.TestCase):
    """Test the ActivationsDataset class."""
    
    def setUp(self):
        """Create a temporary test dataset."""
        self.test_data = [
            {
                "statement": "Test statement 1",
                "label": 1,
                "activations": [0.1, 0.2, 0.3, 0.4],
                "model": "test-model"
            },
            {
                "statement": "Test statement 2", 
                "label": 0,
                "activations": [0.5, 0.6, 0.7, 0.8],
                "model": "test-model"
            }
        ]
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_file.name)
    
    def test_dataset_loading(self):
        """Test that dataset loads correctly."""
        dataset = ActivationsDataset(self.temp_file.name)
        self.assertEqual(len(dataset), 2)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = ActivationsDataset(self.temp_file.name)
        activations, label = dataset[0]
        
        self.assertIsInstance(activations, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(activations.shape[0], 4)
        self.assertEqual(label.item(), 1.0)
    
    def test_invalid_file(self):
        """Test handling of invalid file paths."""
        dataset = ActivationsDataset("nonexistent_file.json")
        self.assertEqual(len(dataset), 0)


class MockClassifier(BaseUncertaintyClassifier):
    """Mock classifier for testing."""
    
    def forward(self, x):
        return torch.sigmoid(torch.sum(x, dim=1, keepdim=True))


class TestBaseClasses(unittest.TestCase):
    """Test the base classes."""
    
    def test_base_classifier(self):
        """Test the base classifier interface."""
        classifier = MockClassifier(input_size=10)
        self.assertEqual(classifier.input_size, 10)
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = classifier(x)
        self.assertEqual(output.shape, (5, 1))


if __name__ == "__main__":
    unittest.main()