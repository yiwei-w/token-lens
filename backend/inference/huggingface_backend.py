# backend/inference/huggingface.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from . import InferenceBackend
import math


class HuggingFaceBackend(InferenceBackend):
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            full_ids = output.sequences[0]
            prompt_len = input_ids.shape[1]
            completion_ids = full_ids[prompt_len:]
            tokens_data = []
            
            for i, token_id in enumerate(completion_ids):
                # Get logits for current token
                logits = output.scores[i][0]  # shape: (vocab_size,)
                
                # Check for NaN values in logits and replace with zeros
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                
                # Convert to log probabilities
                log_softmax = torch.log_softmax(logits, dim=-1)
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get probability and log probability for the selected token
                token_prob = probs[token_id].item()
                log_prob = log_softmax[token_id].item()
                
                # Calculate entropy: -sum(p * log(p))
                # Handle potential NaN in entropy calculation
                entropy = -torch.sum(probs * log_softmax).item()
                if math.isnan(entropy):
                    entropy = 0.0
                
                # Get top-k tokens
                topk_values, topk_indices = torch.topk(logits, k=top_k)
                
                # Get both raw logits and log probabilities for top-k tokens
                topk_logits = topk_values.tolist()
                topk_log_probs = torch.log_softmax(topk_values, dim=-1).tolist()
                
                # Decode top-k tokens
                top_tokens = [self.tokenizer.decode([tid]).strip() for tid in topk_indices.tolist()]
                
                # Ensure no NaN values in the final data
                tokens_data.append({
                    "text": self.tokenizer.decode([token_id]).strip(),
                    "prob": 0.0 if math.isnan(token_prob) else token_prob,
                    "log_prob": 0.0 if math.isnan(log_prob) else log_prob,
                    "entropy": entropy,
                    "top_tokens": top_tokens,
                    "top_logprobs": [0.0 if math.isnan(lp) else lp for lp in topk_log_probs],
                    "top_logits": [0.0 if math.isnan(l) else l for l in topk_logits]
                })
                
            return {
                "full_text": self.tokenizer.decode(full_ids, skip_special_tokens=True),
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
        """Generate a tree of possible completions using HuggingFace transformers."""
        try:
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
            
            # Queue for exploration: (node_id, current_prompt, depth)
            exploration_queue = [(0, prompt, 0)]
            
            while exploration_queue:
                current_node_id, current_prompt, current_depth = exploration_queue.pop(0)
                
                if current_depth >= max_depth:
                    continue
                
                # Tokenize current prompt
                inputs = self.tokenizer(current_prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Get next token logits
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, -1]  # Last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Apply top_k filtering
                if top_k > 0:
                    topk_values, topk_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    filtered_logits[topk_indices] = topk_values
                    logits = filtered_logits
                
                # Apply top_p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply min_p filtering
                if min_p > 0.0:
                    max_logit = torch.max(logits)
                    min_logit_threshold = max_logit + math.log(min_p)
                    logits[logits < min_logit_threshold] = float('-inf')
                
                # Get valid tokens
                valid_mask = logits != float('-inf')
                if not valid_mask.any():
                    continue
                
                valid_indices = torch.where(valid_mask)[0]
                valid_logits = logits[valid_indices]
                valid_probs = torch.softmax(valid_logits, dim=-1)
                valid_log_probs = torch.log_softmax(valid_logits, dim=-1)
                
                # Calculate entropy
                entropy = -torch.sum(valid_probs * valid_log_probs).item()
                
                # Create child nodes for valid tokens
                current_node = tree_nodes[current_node_id]
                for i, token_idx in enumerate(valid_indices):
                    token_id = token_idx.item()
                    token_text = self.tokenizer.decode([token_id])
                    token_prob = valid_probs[i].item()
                    token_log_prob = valid_log_probs[i].item()
                    
                    child_node = {
                        "id": node_id_counter,
                        "parent_id": current_node_id,
                        "text": token_text,
                        "prob": float(token_prob),
                        "log_prob": float(token_log_prob),
                        "entropy": float(entropy),
                        "depth": current_depth + 1,
                        "cumulative_text": current_node["cumulative_text"] + token_text,
                        "full_prompt": current_prompt + token_text,
                        "children": []
                    }
                    
                    tree_nodes.append(child_node)
                    current_node["children"].append(node_id_counter)
                    
                    # Add to exploration queue
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