# backend/inference/vllm_backend.py
import numpy as np
import base64
from vllm import LLM, SamplingParams
from . import InferenceBackend
from typing import Dict, List, Any


class VLLMBackend(InferenceBackend):
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # Initialize vLLM with the model
        self.llm = LLM(model=self.model_name, trust_remote_code=True, max_logprobs=151936)
        # Get the tokenizer from the LLM for token decoding
        self.tokenizer = self.llm.get_tokenizer()
        # assert self.tokenizer.vocab_size == 151936, f"Vocab size is not 151643, but {self.tokenizer.vocab_size}"
        
    def get_token(self, token_id):
        # This is a direct approach - get the raw string representation of the token
        text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        return text
    
    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=-1,
            logprobs=500,  # Request logprobs for top 500 tokens as approximation for entropy; all tokens would be too slow
        )
        
        try:
            # Generate with vLLM
            outputs = self.llm.generate(prompt, sampling_params)
            output = outputs[0]  # Get the first (and only) output
            
            # Extract generated text and token details
            generated_text = output.outputs[0].text
            
            # Process token-level information
            tokens_data = []
            
            # Check if logprobs are available
            if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                # Get the actual token IDs that were generated
                generated_token_ids = output.outputs[0].token_ids
                
                for i, token_logprobs in enumerate(output.outputs[0].logprobs):
                    if not token_logprobs:  # Skip if no logprobs for this token
                        continue
                    
                    float_logprobs = {}
                    for token_id, logprob in token_logprobs.items():
                        float_logprobs[token_id] = logprob.logprob
                    
                    sorted_logprobs = sorted(float_logprobs.items(), key=lambda x: x[1], reverse=True)

                    probs_sum = sum(np.exp(logprob) for logprob in float_logprobs.values())
                    if probs_sum < 0.95:
                        # Handle the case where probability mass is too low
                        print(f"Warning: Sum of probabilities is only {probs_sum:.4f}, less than 0.95")
                    
                    # Get the actual token that was sampled by the model
                    if i < len(generated_token_ids):
                        selected_token_id = generated_token_ids[i]
                        selected_token_text = self.get_token(selected_token_id)
                    
                    
                    
                    # Extract top tokens and their logprobs
                    top_tokens = []
                    top_logprobs_values = []
                    
                    for token_id, logprob in sorted_logprobs[:top_k]:
                        # Use our safe decoding method for all tokens
                        decoded_token = self.get_token(token_id)
                        top_tokens.append(decoded_token)
                        top_logprobs_values.append(logprob)
                    
                    # Calculate probability and entropy
                    logprobs_array = np.array([lp for _, lp in sorted_logprobs])
                    probs = np.exp(logprobs_array)
                    probs = probs / np.sum(probs)  # Normalize to ensure they sum to 1
                    
                    # Get probability of the selected token
                    selected_token_logprob = float_logprobs[selected_token_id]
                    selected_token_prob = np.exp(selected_token_logprob)
                    
                    # Calculate entropy: -sum(p * log(p))
                    # Use a numerically stable approach
                    entropy = -np.sum(probs * logprobs_array)
                    
                    # Estimate logits from logprobs (approximate since we don't have direct access)
                    # This is an approximation assuming temperature=1.0
                    top_logits_values = [lp * 1.0 for lp in top_logprobs_values]  # Simple scaling for demonstration
                    
                    tokens_data.append({
                        "text": selected_token_text,
                        "prob": float(selected_token_prob),
                        "log_prob": float(selected_token_logprob),
                        "entropy": float(entropy),
                        "top_tokens": top_tokens,
                        "top_logprobs": top_logprobs_values,
                        "top_logits": top_logits_values
                    })
            
            return {
                "full_text": generated_text,
                "tokens": tokens_data
            }
            
        except Exception as e:
            # Handle any errors during generation
            error_msg = str(e)
            return {
                "full_text": f"Error during generation: {error_msg}",
                "tokens": []
            }
    
    def generate_tree(self, prompt: str, temperature: float, max_depth: int, top_k: int, top_p: float, min_p: float) -> dict:
        """Generate a tree of possible completions by exploring multiple paths at each step."""
        try:
            tree_nodes = []
            node_id_counter = 0
            
            # Root node represents the initial prompt
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
            
            # Queue for breadth-first exploration: (node_id, current_prompt, depth)
            exploration_queue = [(0, prompt, 0)]
            
            while exploration_queue and max(node["depth"] for node in tree_nodes) < max_depth:
                current_node_id, current_prompt, current_depth = exploration_queue.pop(0)
                
                if current_depth >= max_depth:
                    continue
                
                # Generate one step from current prompt
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=1,  # Generate only one token at a time
                    top_k=-1,
                    top_p=top_p,
                    logprobs=500,  # Get logprobs for entropy calculation
                )
                
                outputs = self.llm.generate(current_prompt, sampling_params)
                output = outputs[0]
                
                if not (hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs and output.outputs[0].logprobs[0]):
                    continue
                
                token_logprobs = output.outputs[0].logprobs[0]
                float_logprobs = {token_id: logprob.logprob for token_id, logprob in token_logprobs.items()}
                
                # Filter tokens based on min_p and top_k
                sorted_logprobs = sorted(float_logprobs.items(), key=lambda x: x[1], reverse=True)
                
                # Apply top_k filtering
                if top_k > 0:
                    sorted_logprobs = sorted_logprobs[:top_k]
                
                # Apply min_p filtering
                max_logprob = sorted_logprobs[0][1] if sorted_logprobs else float('-inf')
                min_logprob_threshold = max_logprob + np.log(min_p)
                filtered_logprobs = [(token_id, logprob) for token_id, logprob in sorted_logprobs if logprob >= min_logprob_threshold]
                
                if not filtered_logprobs:
                    continue
                
                # Calculate entropy for this position
                logprobs_array = np.array([lp for _, lp in filtered_logprobs])
                probs = np.exp(logprobs_array)
                probs = probs / np.sum(probs)  # Normalize
                entropy = -np.sum(probs * logprobs_array)
                
                # Create child nodes for each valid token
                current_node = tree_nodes[current_node_id]
                for i, (token_id, logprob) in enumerate(filtered_logprobs):
                    token_text = self.get_token(token_id)
                    token_prob = np.exp(logprob)
                    
                    child_node = {
                        "id": node_id_counter,
                        "parent_id": current_node_id,
                        "text": token_text,
                        "prob": float(token_prob),
                        "log_prob": float(logprob),
                        "entropy": float(entropy),
                        "depth": current_depth + 1,
                        "cumulative_text": current_node["cumulative_text"] + token_text,
                        "full_prompt": current_prompt + token_text,
                        "children": []
                    }
                    
                    tree_nodes.append(child_node)
                    current_node["children"].append(node_id_counter)
                    
                    # Add to exploration queue for next level
                    exploration_queue.append((node_id_counter, current_prompt + token_text, current_depth + 1))
                    node_id_counter += 1
            
            return {
                "tree_nodes": tree_nodes,
                "max_depth_reached": max(node["depth"] for node in tree_nodes),
                "total_nodes": len(tree_nodes)
            }
            
        except Exception as e:
            return {
                "tree_nodes": [],
                "max_depth_reached": 0,
                "total_nodes": 0,
                "error": str(e)
            }