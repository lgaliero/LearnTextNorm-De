import os
import json
import ollama
import argparse
import configs
import subprocess
from ollama import Client
from typing import Iterable, Tuple, Optional, Dict, List
from configs import Paths, ApiConfig

api_key = os.getenv("OLLAMA_API_KEY")
client = Client(
    host=ApiConfig.HOST,
    headers={'Authorization': 'Bearer ' + api_key}
)

Example = Tuple[str, str]


def load_examples_json(json_path: str) -> List[Dict]:
    """Load examples from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def find_examples_for_sentence(sentence: str, examples_data: List[Dict]) -> Optional[Dict]:
    """Find the example entry matching the given test sentence."""
    for entry in examples_data:
        if entry['test_source'] == sentence:
            return entry
    return None


def query_model(
    sentence: str,
    mode: str,
    examples: Iterable[Example] | None = None,
    baseline_output: str | None = None,
) -> str:
    system_context = (
        ApiConfig.SYS_BASELINE if mode == "baseline"
        else ApiConfig.SYS_2SHOT
    )

    messages = [{"role": "system", "content": system_context}]

    if mode == "2-shot":
        if not examples or len(examples) < 2:
            raise ValueError("2-shot mode requires at least 2 examples")

        # Add the two example pairs as conversation history
        for src, tgt in examples:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": tgt})

        # Add the final user message with baseline and source
        if baseline_output and baseline_output != "to be added":
            user_content = f"Previous attempt: {baseline_output}\n\nCorrect this: {sentence}"
        else:
            user_content = sentence
    else:
        user_content = sentence

    messages.append({
        "role": "user",
        "content": user_content
    })

    response = ""
    for part in client.chat(
        model=ApiConfig.MODEL,
        messages=messages,
        stream=True,
    ):
        if part.message.content:
            response += part.message.content

    return response.strip()


def append_to_tgt(text: str, mode: str):
    if mode == "baseline":
        path = Paths.LLM_BASE
    elif mode == "2-shot":
        path = Paths.LLM_2S
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Replace all newlines and multiple spaces with single space
    single_line = ' '.join(text.split())

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a+", encoding="utf-8") as f:
        f.write(single_line + "\n")


def get_examples_interactively() -> list[Example]:
    """Prompt user to input two example pairs"""
    examples = []
    
    print("\n--- Enter 2 example pairs ---")
    for i in range(2):
        print(f"\nExample {i+1}:")
        src = input(f"  Source {i+1} > ").strip()
        tgt = input(f"  Target {i+1} > ").strip()
        
        if not src or not tgt:
            print("  Warning: Empty example, skipping...")
            continue
            
        examples.append((src, tgt))
    
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "2-shot", "2-shot-json"],
        default=ApiConfig.MODE,
        help="Inference mode",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=Paths.JSON,
        help="Path to examples JSON file (for 2-shot-json mode)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input file with source sentences (one per line)",
    )
    args = parser.parse_args()

    mode = args.mode
    print(f"Running in {mode} mode\n")
    
    # Load examples JSON if using 2-shot-json mode
    examples_data = []
    if mode == "2-shot-json":
        examples_data = load_examples_json(args.json)
        if not examples_data:
            print("Error: Could not load examples JSON. Exiting.")
            exit(1)
        print(f"✓ Loaded {len(examples_data)} examples from {args.json}\n")
    
    if mode == "baseline":
        print("Paste a sentence and press Enter.")
    elif mode == "2-shot":
        print("For each source sentence, you'll provide:")
        print("  1. Two example pairs (source/target)")
        print("  2. Baseline AI output")
        print("  3. Source sentence to correct")
    elif mode == "2-shot-json":
        print("Paste a test source sentence and the system will:")
        print("  1. Find matching examples from JSON")
        print("  2. Retrieve baseline output")
        print("  3. Generate corrected output")
    
    print("Type :q to quit.\n")

    # Determine input source
if args.input:
    # File-based processing
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found.")
        exit(1)
    
    print(f"Processing {len(sentences)} sentences from {args.input}\n")
    
    for idx, sentence in enumerate(sentences, 1):
        print(f"[{idx}/{len(sentences)}] Processing: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
        
        examples = None
        baseline_output = None
        
        if mode == "2-shot-json":
            entry = find_examples_for_sentence(sentence, examples_data)
            
            if not entry:
                print(f"  ⚠️  Warning: No examples found, skipping sentence.")
                continue
            
            examples = [
                (ex['source'], ex['target']) 
                for ex in entry['examples'][:2]
            ]
            baseline_output = entry.get('baseline_output', 'to be added')
        
        try:
            output = query_model(
                sentence,
                mode if mode != "2-shot-json" else "2-shot",
                examples=examples,
                baseline_output=baseline_output
            )
            
            print(f"  ✓ Generated output")
            append_to_tgt(output, "2-shot" if mode == "2-shot-json" else mode)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Completed processing {len(sentences)} sentences")

else:
    # Interactive mode (original behavior)
    while True:
        examples = None
        baseline_output = None
        
        # Get the source sentence first
        sentence = input("\nSRC > ").strip()
        
        if sentence == ":q":
            print("Bye 👋")
            break
        
        if not sentence:
            continue
        
        if mode == "2-shot-json":
            # Find examples from JSON
            entry = find_examples_for_sentence(sentence, examples_data)
            
            if not entry:
                print(f"⚠️  Warning: No examples found for this sentence in JSON.")
                print("   The sentence might not be in the test set or JSON wasn't generated correctly.")
                
                response = input("   Continue with manual input? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    continue
                
                # Fall back to manual input
                examples = get_examples_interactively()
                if len(examples) < 2:
                    print("Error: Need exactly 2 examples. Skipping...\n")
                    continue
                
                baseline_output = input("\nTGT_BASELINE > ").strip()
            else:
                # Extract examples from JSON
                examples = [
                    (ex['source'], ex['target']) 
                    for ex in entry['examples'][:2]  # Take first 2 examples
                ]
                
                baseline_output = entry.get('baseline_output', 'to be added')
                
                # Display what was found
                print(f"\n✓ Found examples:")
                for i, (src, tgt) in enumerate(examples, 1):
                    dataset = entry['examples'][i-1].get('dataset', 'unknown')
                    score = entry['examples'][i-1]
                    print(f"  Example {i}:")
                    print(f"    SRC: {src[:80]}{'...' if len(src) > 80 else ''}")
                    print(f"    TGT: {tgt[:80]}{'...' if len(tgt) > 80 else ''}")
                
                if baseline_output != "to be added":
                    print(f"\n✓ Baseline: {baseline_output[:80]}{'...' if len(baseline_output) > 80 else ''}")
                else:
                    print(f"\n⚠️  No baseline available for this sentence")
                
                print()
        
        elif mode == "2-shot":
            # Manual mode: get examples interactively
            examples = get_examples_interactively()
            
            if len(examples) < 2:
                print("Error: Need exactly 2 examples. Try again.\n")
                continue
            
            # Get baseline output
            baseline_output = input("\nTGT_BASELINE > ").strip()
            if not baseline_output:
                print("Warning: 2-shot mode requires baseline output. Skipping...\n")
                continue

        try:
            output = query_model(
                sentence,
                mode if mode != "2-shot-json" else "2-shot",
                examples=examples,
                baseline_output=baseline_output
            )
            
            print("\nTGT >", output, "\n")
            append_to_tgt(output, "2-shot" if mode == "2-shot-json" else mode)
            
        except Exception as e:
            print(f"Error: {e}\n")