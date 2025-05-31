"""
Unit tests for model architectures and utilities.
"""
import unittest
import torch
import tempfile
import os
from src.model import ISUCClassifier, SimpleMLPClassifier, create_classifier, load_classifier


class TestModelArchitectures(unittest.TestCase):
    """Test different classifier architectures."""
    
    def test_isuc_classifier(self):
        """Test ISUC classifier architecture."""
        input_size = 128
        classifier = ISUCClassifier(input_size)
        
        # Test forward pass
        x = torch.randn(10, input_size)
        output = classifier(x)
        
        self.assertEqual(output.shape, (10, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Sigmoid output
    
    def test_simple_mlp_classifier(self):
        """Test Simple MLP classifier architecture."""
        input_size = 256
        hidden_dims = [512, 256]
        classifier = SimpleMLPClassifier(input_size, hidden_dims)
        
        # Test forward pass
        x = torch.randn(5, input_size)
        output = classifier(x)
        
        self.assertEqual(output.shape, (5, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Sigmoid output
    
    def test_create_classifier_factory(self):
        """Test classifier factory function."""
        input_size = 100
        
        # Test ISUC creation
        isuc = create_classifier("isuc", input_size)
        self.assertIsInstance(isuc, ISUCClassifier)
        
        # Test Simple MLP creation
        mlp = create_classifier("simple_mlp", input_size, hidden_dims=[64, 32])
        self.assertIsInstance(mlp, SimpleMLPClassifier)
        
        # Test invalid classifier type
        with self.assertRaises(ValueError):
            create_classifier("invalid_type", input_size)
    
    def test_load_classifier(self):
        """Test loading classifier from checkpoint."""
        input_size = 50
        
        # Create and save a model
        original_model = ISUCClassifier(input_size)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(original_model.state_dict(), f.name)
            
            # Load the model
            loaded_model = load_classifier("isuc", input_size, f.name)
            
            # Test that the models have the same weights
            x = torch.randn(3, input_size)
            original_output = original_model(x)
            loaded_output = loaded_model(x)
            
            self.assertTrue(torch.allclose(original_output, loaded_output))
        
        # Clean up
        os.unlink(f.name)
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading from nonexistent checkpoint."""
        input_size = 30
        model = load_classifier("isuc", input_size, "nonexistent_checkpoint.pt")
        
        # Should still create a model even if checkpoint doesn't exist
        self.assertIsInstance(model, ISUCClassifier)


if __name__ == "__main__":
    unittest.main()