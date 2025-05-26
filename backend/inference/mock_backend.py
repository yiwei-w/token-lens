# backend/inference/mock.py
from . import InferenceBackend
from transformers import AutoTokenizer
import random
import numpy as np


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
            
            # Convert logits to log probabilities
            # First, create a numpy array for numerical stability
            logits_array = np.array(top_logits)
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits_array - np.max(logits_array))  # Subtract max for numerical stability
            probs = exp_logits / np.sum(exp_logits)
            # Take log to get log probabilities
            top_logprobs = np.log(probs).tolist()
            
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
                "top_logprobs": top_logprobs,
                "top_logits": top_logits
            }
            tokens_data.append(token_info)
        
        # Join the tokens with spaces (you might want to adjust this based on the tokenizer's behavior)
        full_text = " ".join(t["text"] for t in tokens_data)
        
        return {
            "full_text": full_text,
            "tokens": tokens_data
        }
    
    def generate_tree(self, prompt: str, temperature: float, max_depth: int, top_k: int, top_p: float, min_p: float) -> dict:
        """Generate a mock tree of possible completions."""
        tree_nodes = []
        node_id_counter = 0
        
        # Root node
        root_node = {
            "id": node_id_counter,
            "parent_id": None,
            "text": "",
            "prob": 1.0,
            "log_prob": 0.0,
            "entropy": 0.0,
            "depth": 0,
            "cumulative_text": "",
            "full_prompt": prompt,
            "children": []
        }
        tree_nodes.append(root_node)
        node_id_counter += 1
        
        # Generate tree levels
        current_level_nodes = [0]  # Start with root node
        
        for depth in range(max_depth):
            next_level_nodes = []
            
            for node_id in current_level_nodes:
                # Generate 2-3 random children for each node
                num_children = random.randint(2, min(top_k, 3))
                current_node = tree_nodes[node_id]
                
                for i in range(num_children):
                    # Random token from vocab
                    token = random.choice(self.vocab_list)
                    # Clean the token
                    try:
                        cleaned_token = self.tokenizer.decode([self.tokenizer.encode(token, add_special_tokens=False)[0]])
                    except:
                        cleaned_token = token
                    
                    child_node = {
                        "id": node_id_counter,
                        "parent_id": node_id,
                        "text": cleaned_token,
                        "prob": random.uniform(0.1, 0.8),
                        "log_prob": random.uniform(-3.0, -0.2),
                        "entropy": random.uniform(1.0, 4.0),
                        "depth": depth + 1,
                        "cumulative_text": current_node["cumulative_text"] + cleaned_token,
                        "full_prompt": current_node["full_prompt"] + cleaned_token,
                        "children": []
                    }
                    
                    tree_nodes.append(child_node)
                    current_node["children"].append(node_id_counter)
                    next_level_nodes.append(node_id_counter)
                    node_id_counter += 1
            
            current_level_nodes = next_level_nodes
            if not current_level_nodes:  # No more nodes to expand
                break
        
        return {
            "tree_nodes": tree_nodes,
            "max_depth_reached": max_depth,
            "total_nodes": len(tree_nodes)
        }