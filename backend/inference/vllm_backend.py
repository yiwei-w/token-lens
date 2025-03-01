# backend/inference/vllm_backend.py
import numpy as np
from vllm import LLM, SamplingParams
from . import InferenceBackend


class VLLMBackend(InferenceBackend):
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # Initialize vLLM with the model
        self.llm = LLM(model=self.model_name, trust_remote_code=True)
        # Get the tokenizer from the LLM for token decoding
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=top_k,
            logprobs=top_k,  # Request logprobs for top_k tokens
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
                for i, token_logprobs in enumerate(output.outputs[0].logprobs):
                    if not token_logprobs:  # Skip if no logprobs for this token
                        continue
                        
                    # Get token text - we'll use the top token from logprobs
                    sorted_logprobs = sorted(token_logprobs.items(), key=lambda x: x[1], reverse=True)
                    token_text = sorted_logprobs[0][0]
                    
                    # Extract top tokens and their logprobs
                    top_tokens = []
                    top_logprobs_values = []
                    
                    for token, logprob in sorted_logprobs[:top_k]:
                        top_tokens.append(token)
                        top_logprobs_values.append(logprob)
                    
                    # Calculate probability and entropy
                    logprobs_array = np.array([lp for _, lp in sorted_logprobs])
                    probs = np.exp(logprobs_array)
                    probs = probs / np.sum(probs)  # Normalize to ensure they sum to 1
                    
                    # Get probability of the selected token
                    token_prob = np.exp(sorted_logprobs[0][1])
                    log_prob = sorted_logprobs[0][1]
                    
                    # Calculate entropy: -sum(p * log(p))
                    # Use a numerically stable approach
                    entropy = -np.sum(probs * logprobs_array)
                    
                    tokens_data.append({
                        "text": token_text,
                        "prob": float(token_prob),
                        "log_prob": float(log_prob),
                        "entropy": float(entropy),
                        "top_tokens": top_tokens,
                        "top_logits": top_logprobs_values  # Keep the field name for compatibility
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