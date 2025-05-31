# Internal States Uncertainty Classifier (ISUC)

**ISUC is a general framework for training uncertainty classifiers using internal activations from Large Language Models.**

With the growing literature on uncertainty and hallucinations, it is increasingly important to compare different methods for evaluating and acting upon uncertainty. Many studies have shown that LLMs encode information about their confidence and the truthfulness of the generated content in their internal states, but this information is not explicitly expressed in the output. This motivates the need to inspect their internal representations.

Such classifiers can be used at **inference time** (e.g., detecting when the LLM is lying) or during **training** (e.g., using ISUC as a reward model). This codebase is modular and extensible, enabling the **reproduction of multiple papers on uncertainty classification**. Pretrained models will be uploaded to Hugging Face for the community to use.

## 🏗️ Architecture

The framework is built with clean abstractions that allow easy extension for different:
- **Datasets**: Add new dataset processors by implementing `BaseDatasetProcessor`
- **Models**: Add new classifier architectures by extending `BaseUncertaintyClassifier`
- **Activation Extractors**: Support different model types by implementing `BaseActivationExtractor`


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
python main.py --classifier_type SAPLMA
```

### 3. Skip Data Processing (if already done)
```bash
python main.py --skip_data_processing --use_wandb
```

## 📊 Available Classifiers

### SAPLMA Classifier
Based on Azaria & Mitchell (2023), designed for uncertainty detection using internal states.
- Architecture: 256 → 128 → 64 → 1 with ReLU and Dropout
- Optimized for LLM activation patterns

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