import os
import time
from typing import Dict, Any, Tuple

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def call_llm(prompt: str, model: str = 'qwen2.5:7b') -> Dict[str, Any]:
    if os.environ.get('MOCK_LLM') == 'true':
        return {'answer': 'Mock answer', 'latency': 0.1}

    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama library not installed. Run: pip install ollama")

    start = time.time()
    try:
        response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        return {
            'answer': response['message']['content'].strip(),
            'latency': time.time() - start
        }
    except Exception as e:
        if 'ConnectionError' in str(type(e)) or 'failed to connect' in str(e).lower():
            raise ConnectionError("Ollama connection failed. Ensure Ollama is running or set MOCK_LLM=true") from e
        raise


def llm_judge(question: str, ground_truth: str, generated_answer: str,
              model: str = 'qwen2.5:7b') -> Tuple[bool, str]:
    if os.environ.get('MOCK_LLM') == 'true':
        is_correct = ground_truth.lower() in generated_answer.lower()
        return is_correct, "Mock judgment"

    prompt = f"""You are evaluating a question-answering system.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Does the generated answer correctly answer the question according to the ground truth?
Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation."""

    try:
        result = call_llm(prompt, model=model)
        judgment = result['answer']
        return judgment.upper().startswith('CORRECT'), judgment
    except Exception as e:
        is_correct = ground_truth.lower() in generated_answer.lower()
        return is_correct, f"Fallback judgment: {e}"