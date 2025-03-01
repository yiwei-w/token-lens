# backend/inference/vllm_backend.py
import numpy as np
from vllm import LLM, SamplingParams
from . import InferenceBackend


class VLLMBackend(InferenceBackend):
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # Initialize vLLM with the model
        self.llm = LLM(model=self.model_name, trust_remote_code=True, max_logprobs=151936)
        # Get the tokenizer from the LLM for token decoding
        self.tokenizer = self.llm.get_tokenizer()
        # assert self.tokenizer.vocab_size == 151936, f"Vocab size is not 151643, but {self.tokenizer.vocab_size}"

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
                        selected_token_text = self.tokenizer.decode([selected_token_id])
                    
                    
                    
                    # Extract top tokens and their logprobs
                    top_tokens = []
                    top_logprobs_values = []
                    
                    for token_id, logprob in sorted_logprobs[:top_k]:
                        decoded_token = self.tokenizer.decode([token_id])
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