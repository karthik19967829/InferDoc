
## LLM-Based Data Creation

InferDoc now includes an LLM-based data creation system for generating synthetic training data.

### Features

- Generate synthetic documents for various document types
- Create question-answer pairs for document inference training
- Support for multiple document categories (legal, medical, financial, etc.)
- Configurable data generation parameters
- JSON output format for easy integration

### Usage

```python
from llm_data_creator import LLMDataCreator

# Initialize with your OpenAI API key
creator = LLMDataCreator(api_key="your-api-key")

# Generate QA pairs for legal documents
qa_pairs = creator.generate_document_qa_pairs("legal", num_pairs=10)

# Generate synthetic documents
documents = creator.generate_synthetic_documents("medical", num_docs=5)

# Create a complete training dataset
stats = creator.create_training_dataset(
    document_types=["legal", "medical", "financial"],
    output_file="training_data.json"
)
```

### Command Line Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Generate training data
python llm_data_creator.py --document-types legal medical financial --output my_dataset.json

# Run example
python examples/llm_data_creation_example.py
```

### Requirements

Install additional dependencies:
```bash
pip install -r requirements_llm.txt
```
