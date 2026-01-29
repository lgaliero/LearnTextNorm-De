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
  
- **`batch_processing.py`** - Sequential batch processing
  - `process_single_sentence()` - Process one sentence with timing/memory diagnostics
  - `process_batch()` - Process multiple sentences sequentially (optimized for `OLLAMA_NUM_PARALLEL=1`)
  
- **`api_query_cli.py`** - CLI wrapper for inference pipeline

### Key Features

- **Sequential Processing** - Optimized for Ollama with `OLLAMA_NUM_PARALLEL=1` (fastest approach)
- **Streaming Responses** - Low-latency token-by-token streaming from LLM
- **Memory Monitoring** - Track memory usage and timing for each sentence
- **Flexible Prompting** - Support for baseline and 2-shot learning modes
- **JSON Example Management** - Load example pairs from structured JSON files

### Usage

**As a CLI:**
```bash
# Baseline mode (zero-shot)
python inference/api_query_cli.py --mode baseline --input test.src --model llama3.2

# 2-shot mode with examples from JSON
python inference/api_query_cli.py --mode 2-shot-json --json examples.json --input test.src --model gpt

# Interactive mode (no input file)
python inference/api_query_cli.py --mode baseline --model llama3.2
```

**As a library:**
```python
from inference import ModelClient, load_examples_json, process_batch

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

### Performance Notes

- **Sequential is faster** when `OLLAMA_NUM_PARALLEL=1` (Ollama default)
- Each sentence takes ~40 seconds with GPT-20B quantized model on CPU
- First sentence includes model loading time (~6.5 minutes for GPT-20B)
- Use HPC array jobs for large batches (see cluster deployment guide)

### Environment Variables

Set these for optimal Ollama performance:
```bash
export OLLAMA_NUM_PARALLEL=1          # Limit concurrent requests
export OLLAMA_MAX_LOADED_MODELS=1     # Keep only one model in memory
export OLLAMA_CONTEXT_SIZE=2048       # Reduce context window for speed
```

---