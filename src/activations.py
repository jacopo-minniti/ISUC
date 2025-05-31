"""
Activation extraction utilities for LLMs.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from src.base import BaseActivationExtractor


class HuggingFaceActivationExtractor(BaseActivationExtractor):
    """Extracts activations from HuggingFace transformer models."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.eval()
        self.model.to(self.device)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        print(f"Loaded {self.model_id} on {self.device}")
    
    def extract_activations(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Extract activations from the specified layer for the last token."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Get activations from the specified layer (layer_idx + 1 due to embedding layer)
            target_idx = layer_idx + 1
            if target_idx < len(outputs.hidden_states):
                layer_activations = outputs.hidden_states[target_idx].detach().cpu()
                # Return activations for the last token
                return layer_activations.squeeze(0)[-1, :]
            else:
                print(f"Layer index {layer_idx} out of bounds for model with {len(outputs.hidden_states)-1} layers")
                return None
        
        except Exception as e:
            print(f"Error extracting activations: {e}")
            return None