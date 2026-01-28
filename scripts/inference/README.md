## Module: Inference (`inference/`)

Handles LLM-based text normalization with support for different prompting strategies.

### Files

- **`model_api.py`** - Core API interaction with LLM
  - `ModelClient` class for managing API connections
  - `query_model()` for single inference calls
  
- **`io_utils.py`** - File I/O for inference outputs
  - `append_to_tgt()` - Save model outputs to appropriate files
  - `load_sentences_from_file()` - Load input sentences
  
- **`prompt_utils.py`** - Example management for few-shot prompting
  - `load_examples_json()` - Load example pairs from JSON
  - `find_examples_for_sentence()` - Find relevant examples
  - `get_examples_interactively()` - Collect examples from user
  
- **`batch_processor.py`** - Parallel/batch processing
  - `process_single_sentence()` - Process one sentence
  - `process_batch()` - Process multiple sentences
  
- **`api_query_cli.py`** - CLI wrapper

### Usage

**As a CLI:**
```bash
# Baseline mode
python inference/api_query_cli.py --mode baseline --input test.src --model llama3.2

# 2-shot mode with examples
python inference/api_query_cli.py --mode 2-shot-json --json examples.json --input test.src

# Interactive mode
python inference/api_query_cli.py --mode baseline
```

**As a library:**
```python
from inference import ModelClient, load_examples_json, process_batch

# Initialize client
client = ModelClient(host="http://localhost:11434")

# Load examples
examples = load_examples_json("prompts.json")

# Process sentences
results = process_batch(
    sentences=["Ich bin gut.", "Das ist schön."],
    mode="baseline",
    model="llama3.2",
    model_client=client,
    system_baseline="You are a helpful assistant.",
    system_2shot="You normalize text.",
    examples_data=examples
)
```

---