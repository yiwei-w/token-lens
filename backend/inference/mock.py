# backend/inference/mock.py
from . import InferenceBackend
from transformers import AutoTokenizer
import random

class MockBackend(InferenceBackend):
    """
    A mock inference backend that samples random tokens from the DeepSeek-R1-Distill-Qwen-1.5B
    tokenizer vocabulary. Useful for testing the frontend without running the actual model.
    """
    def __init__(self):
        # Load just the tokenizer from the model
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        
        # Get the vocabulary but filter out special tokens
        vocab = self.tokenizer.get_vocab()
        self.vocab_list = [token for token in vocab.keys() 
                          if not token.startswith('<') and not token.endswith('>')]
        
    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        tokens_data = []
        
        for _ in range(max_new_tokens):
            # Randomly sample top_k tokens
            sampled_tokens = random.sample(self.vocab_list, min(top_k, len(self.vocab_list)))
            
            # Clean up the tokens by removing special characters but preserve spaces
            cleaned_tokens = [self.tokenizer.decode([self.tokenizer.encode(t, add_special_tokens=False)[0]]) 
                            for t in sampled_tokens]
            
            # Generate random logits for the top tokens (higher values for first tokens)
            top_logits = [random.uniform(2.0, 5.0) for _ in range(len(cleaned_tokens))]
            top_logits.sort(reverse=True)  # Sort in descending order
            
            # Calculate probability from first logit
            prob = random.uniform(0.3, 0.9)
            log_prob = random.uniform(-2.0, -0.1)
            entropy = random.uniform(1.0, 4.0)
            
            token_info = {
                "text": cleaned_tokens[0],
                "prob": prob,
                "log_prob": log_prob,
                "entropy": entropy,
                "top_tokens": cleaned_tokens,
                "top_logits": top_logits
            }
            tokens_data.append(token_info)
        
        # Join the tokens with spaces (you might want to adjust this based on the tokenizer's behavior)
        full_text = " ".join(t["text"] for t in tokens_data)
        
        return {
            "full_text": full_text,
            "tokens": tokens_data
        }