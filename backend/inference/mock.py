# backend/inference/mock.py
from . import InferenceBackend

class MockBackend(InferenceBackend):
    """
    A mock inference backend that returns a fixed list of tokens, probabilities,
    and top-k logits. Useful for testing the frontend quickly without running
    an actual model.
    """
    def __init__(self):
        # Define a static list of token data
        self.mock_tokens = [
            {
                "text": "Hello",
                "prob": 0.9,
                "log_prob": -0.1053605,
                "entropy": 1.5,
                "top_tokens": ["Hello", "Hi", "Hey"],
                "top_logits": [5.0, 3.2, 2.9]
            },
            {
                "text": "world",
                "prob": 0.8,
                "log_prob": -0.2231435,
                "entropy": 2.2,
                "top_tokens": ["world", "earth", "universe"],
                "top_logits": [4.8, 3.5, 2.7]
            },
            {
                "text": "this",
                "prob": 0.5,
                "log_prob": -0.693147,
                "entropy": 3.0,
                "top_tokens": ["this", "that", "it"],
                "top_logits": [2.5, 2.3, 2.1]
            },
            {
                "text": "is",
                "prob": 0.6,
                "log_prob": -0.51,
                "entropy": 2.8,
                "top_tokens": ["is", "was", "be"],
                "top_logits": [3.1, 2.7, 2.2]
            },
            {
                "text": "a",
                "prob": 0.4,
                "log_prob": -0.9162907,
                "entropy": 3.5,
                "top_tokens": ["a", "the", "one"],
                "top_logits": [1.8, 1.5, 1.2]
            },
            {
                "text": "mock",
                "prob": 0.7,
                "log_prob": -0.3566749,
                "entropy": 2.1,
                "top_tokens": ["mock", "fake", "dummy"],
                "top_logits": [3.5, 2.6, 2.0]
            },
            {
                "text": "inference",
                "prob": 0.65,
                "log_prob": -0.4307829,
                "entropy": 2.4,
                "top_tokens": ["inference", "prediction", "guess"],
                "top_logits": [3.0, 2.8, 2.1]
            },
            {
                "text": "backend",
                "prob": 0.55,
                "log_prob": -0.5978370,
                "entropy": 2.7,
                "top_tokens": ["backend", "server", "system"],
                "top_logits": [2.8, 2.4, 2.1]
            },
            {
                "text": "!",
                "prob": 0.95,
                "log_prob": -0.051293,
                "entropy": 1.0,
                "top_tokens": ["!", ".", "?" ],
                "top_logits": [5.2, 4.9, 4.0]
            }
        ]

    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        # Return as many tokens as requested, up to the length of our mock list
        n = min(max_new_tokens, len(self.mock_tokens))
        tokens_data = self.mock_tokens[:n]
        full_text = " ".join(t["text"] for t in tokens_data)
        return {
            "full_text": full_text,
            "tokens": tokens_data
        }