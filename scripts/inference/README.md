## Module: Inference (`inference/`)

Handles LLM-based text normalization with support for different prompting strategies.

### Files

- **`model_api.py`** - Core API interaction with LLM
  - `ModelClient` class for managing API connections
  - `query_model()` for single inference calls with streaming
  
- **`io_utils.py`** - File I/O for inference outputs
  - `append_to_tgt()` - Save model outputs to appropriate files
  - `load_sentences_from_file()` - Load input sentences
  
- **`prompt_utils.py`** - Example management for few-shot prompting
  - `load_examples_json()` - Load example pairs from JSON
  - `find_examples_for_sentence()` - Find relevant examples for a sentence
  - `extract_examples_from_entry()` - Extract example pairs from an entry
  - `get_examples_interactively()` - Collect examples from user input
  
  
### Usage

**As a library:**
```python
from inference import ModelClient, load_examples_json

# Initialize client
client = ModelClient(host="http://localhost:11434")

# Load examples (for 2-shot mode)
examples_data = load_examples_json("prompts.json")

# Results is a list of (idx, sentence, output) tuples
for idx, sentence, output in results:
    print(f"{idx}: {sentence} → {output}")
```

**Processing individual sentences:**
```python
from inference import ModelClient

client = ModelClient(host="http://localhost:11434")

# Single inference call
output = client.query_model(
    sentence="Ich bin gut",
    mode="baseline",
    model="llama3.2",
    system_baseline="Normalize this text.",
    system_2shot="",
    examples=None,
    baseline_output=None
)

print(output)  # Normalized text
```

**Last Updated:** 29th January 2026  
**Maintainer:** Lucia Galiero