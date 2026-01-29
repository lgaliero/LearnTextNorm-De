"""
Core API interaction logic for LLM inference.
"""

import os
from typing import Iterable, Tuple, Optional
from ollama import Client

Example = Tuple[str, str]


class ModelClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self, host: str, api_key: str = None):
        """Initialize the model client."""
        if api_key is None:
            api_key = os.getenv("OLLAMA_API_KEY")
        
        self.client = Client(
            host=host,
            headers={'Authorization': 'Bearer ' + api_key} if api_key else {}
        )
    
    def query_model(
        self,
        sentence: str,
        mode: str,
        model: str,
        system_baseline: str,
        system_2shot: str,
        examples: Optional[Iterable[Example]] = None,
        baseline_output: Optional[str] = None,
    ) -> str:
        """
        Query the model with a sentence.
        
        Args:
            sentence: Input sentence to normalize
            mode: Inference mode ('baseline' or '2-shot-json')
            model: Model identifier
            system_baseline: System prompt for baseline mode
            system_2shot: System prompt for 2-shot mode
            examples: Example pairs for 2-shot mode
            baseline_output: Previous baseline output for refinement
            
        Returns:
            Model's response as a string
        """
        system_context = (
            system_baseline if mode == "baseline"
            else system_2shot
        )

        messages = [{"role": "system", "content": system_context}]

        if mode == "2-shot-json":
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

        # Use streaming like old code (fastest for Ollama)
        response = ""
        for part in self.client.chat(
            model=model,  # ← FIXED: Use the model parameter, not hardcoded
            messages=messages,
            stream=True,
        ):
            if part.message.content:
                response += part.message.content

        return response.strip()