# Uncertainty Classifiers using LLM Internal States (ISUC)

A general framework for training uncertainty classifiers using internal activations from Large Language Models. This codebase is designed to be modular and extensible for reproducing multiple papers on uncertainty classification.

## 🏗️ Architecture

The framework is built with clean abstractions that allow easy extension for different:
- **Datasets**: Add new dataset processors by implementing `BaseDatasetProcessor`
- **Models**: Add new classifier architectures by extending `BaseUncertaintyClassifier`
- **Activation Extractors**: Support different model types by implementing `BaseActivationExtractor`

### Project Structure

```
src/
├── base.py              # Base classes and interfaces
├── activations.py       # Activation extraction from LLMs
├── data_processors.py   # Dataset-specific processors
├── model.py            # Classifier architectures
├── train.py            # Training utilities
├── evaluate.py         # Evaluation metrics
└── utils.py            # Utility functions

test/
├── test_base.py        # Tests for base classes
├── test_models.py      # Tests for model architectures
├── classify_query.py   # Single query classification
├── sample_test_set.py  # Sample evaluation utility
└── run_tests.py        # Test runner

examples.py             # Usage examples
main.py                 # Main training script
```

## 🚀 Quick Start

### 1. Basic Training
```bash
# Train with default settings (ISUC classifier on true-false dataset)
python main.py --use_wandb

# Train with custom settings
python main.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "true-false" \
    --classifier_type "isuc" \
    --layer_idx 16 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 10 \
    --use_wandb
```

### 2. Different Classifier Types
```bash
# Train ISUC classifier (Azaria & Mitchell, 2023)
python main.py --classifier_type isuc

# Train Simple MLP classifier
python main.py --classifier_type simple_mlp
```

### 3. Skip Data Processing (if already done)
```bash
python main.py --skip_data_processing --use_wandb
```

## 📊 Available Classifiers

### ISUC Classifier
Based on Azaria & Mitchell (2023), designed for uncertainty detection using internal states.
- Architecture: 256 → 128 → 64 → 1 with ReLU and Dropout
- Optimized for LLM activation patterns

### Simple MLP Classifier
Configurable multi-layer perceptron for experimentation.
- Customizable hidden dimensions
- Good baseline for comparison

## 🔧 Adding New Components

### Adding a New Dataset
```python
from src.base import BaseDatasetProcessor
from src.activations import HuggingFaceActivationExtractor

class MyDatasetProcessor(BaseDatasetProcessor):
    def process_dataset(self, model_id: str, layer_idx: int, output_dir: str) -> bool:
        # Implement your dataset processing logic
        pass
```

### Adding a New Classifier
```python
from src.base import BaseUncertaintyClassifier

class MyClassifier(BaseUncertaintyClassifier):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement your classifier architecture
        pass
```

## 🧪 Testing

Run all tests:
```bash
python test/run_tests.py
```

Test a single query:
```bash
python test/classify_query.py "The Earth is flat"
```

Evaluate on a sample:
```bash
python test/sample_test_set.py ./models/isuc-v05-26.pt isuc ./data/true-false/prepared/test.json 100
```

## 🛠️ Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--dataset` | `true-false` | Dataset name |
| `--classifier_type` | `isuc` | Classifier architecture |
| `--layer_idx` | `16` | Layer for activation extraction |
| `--batch_size` | `64` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--epochs` | `5` | Number of training epochs |
| `--use_wandb` | `False` | Enable W&B logging |
| `--skip_data_processing` | `False` | Skip data preparation |