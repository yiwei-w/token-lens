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
            top_k=-1,
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
                    
                    # Convert Logprob objects to float values if needed
                    float_logprobs = {}
                    for token, logprob in token_logprobs.items():
                        # Check if logprob is a Logprob object
                        if hasattr(logprob, 'logprob'):
                            # It's a Logprob object, extract the logprob value
                            float_logprobs[token] = logprob.logprob
                        else:
                            # It's already a float or something else
                            float_logprobs[token] = float(logprob)
                    
                    # Sort using the float values
                    sorted_logprobs = sorted(float_logprobs.items(), key=lambda x: x[1], reverse=True)
                    
                    # Get token ID and decode it to text
                    token_id = sorted_logprobs[0][0]
                    # Check if token_id is a string representation of an integer
                    if isinstance(token_id, str) and token_id.isdigit():
                        token_id = int(token_id)
                    # Decode the token ID to get the actual text
                    try:
                        token_text = self.tokenizer.decode([token_id]) if isinstance(token_id, int) else token_id
                    except:
                        # Fallback if decoding fails
                        token_text = str(token_id)
                    
                    # Extract top tokens and their logprobs
                    top_tokens = []
                    top_logprobs_values = []
                    
                    for token, logprob in sorted_logprobs[:top_k]:
                        # Decode token if it's an ID
                        if isinstance(token, str) and token.isdigit():
                            token = int(token)
                        try:
                            decoded_token = self.tokenizer.decode([token]) if isinstance(token, int) else token
                            top_tokens.append(decoded_token)
                        except:
                            top_tokens.append(str(token))
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
                    
                    # Estimate logits from logprobs (approximate since we don't have direct access)
                    # This is an approximation assuming temperature=1.0
                    top_logits_values = [lp * 1.0 for lp in top_logprobs_values]  # Simple scaling for demonstration
                    
                    tokens_data.append({
                        "text": token_text,
                        "prob": float(token_prob),
                        "log_prob": float(log_prob),
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