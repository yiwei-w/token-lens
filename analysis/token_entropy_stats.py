import random
import numpy as np
import json
from collections import defaultdict
from vllm import LLM, SamplingParams
import argparse
import os
import heapq
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison dataset and analyze token entropy")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                        help="Model to use for generation")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of comparison samples to generate")
    parser.add_argument("--samples_per_prompt", type=int, default=5,
                        help="Number of sequences to generate for each prompt")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default="entropy_stats_output", 
                        help="Base directory for output files")
    parser.add_argument("--top_n", type=int, default=100,
                        help="Number of top entropy tokens to display")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--template_names", type=str, default=None,
                        help="Comma-separated list of template names to use (default: all)")
    return parser.parse_args()

def get_prompt_templates():
    """
    Define prompt templates directly in the code.
    
    Returns:
        Dictionary mapping template names to prompt formats
    """
    templates = {
        "boxed_add": "Compare sqrt({a}) + sqrt({b}) and sqrt({c}) + sqrt({d}). Which is larger? "
                      "Please reason step by step, and put the larger number within \\boxed{{}}.\n<think>\n",
        
        "boxed_sub": "Compare sqrt({a}) - sqrt({b}) and sqrt({c}) - sqrt({d}). Which is larger? "
                       "Please reason step by step, and put the larger number within \\boxed{{}}.\n<think>\n",
        
        "boxed_add_no_approx": "Without any numerical approximation, compare sqrt({a}) + sqrt({b}) and sqrt({c}) + sqrt({d}). "
                               "Please reason step by step, and put the larger number within \\boxed{{}}.\n<think>\n",
        
        "boxed_sub_no_approx": "Without any numerical approximation, compare sqrt({a}) - sqrt({b}) and sqrt({c}) - sqrt({d}). "
                               "Please reason step by step, and put the larger number within \\boxed{{}}.\n<think>\n",
        "add": "Compare sqrt({a}) + sqrt({b}) and sqrt({c}) + sqrt({d}). Which is larger?\n<think>\n",
        "sub": "Compare sqrt({a}) - sqrt({b}) and sqrt({c}) - sqrt({d}). Which is larger?\n<think>\n",
        "add_no_approx": "Without any numerical approximation, compare sqrt({a}) + sqrt({b}) and sqrt({c}) + sqrt({d}).\n<think>\n",
        "sub_no_approx": "Without any numerical approximation, compare sqrt({a}) - sqrt({b}) and sqrt({c}) - sqrt({d}).\n<think>\n"

    }
    return templates

def generate_comparison_samples(template, num_samples=200, random_seed=42):
    """
    Generate samples of 4 different non-zero integers from 1-50.
    Group the first and fourth smallest in one group and the middle two in another.
    Ensures no duplicate samples are generated.
    
    Args:
        template: String template for prompts
        num_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        List of prompts with comparison questions
    """
    random.seed(random_seed)
    
    # Check if we can generate enough unique samples
    max_possible_samples = 50 * 49 * 48 * 47 // 24  # C(50,4)
    num_samples = min(num_samples, max_possible_samples)
    
    samples = set()
    
    # Keep generating samples until we have the required number of unique samples
    while len(samples) < num_samples:
        # Sample 4 different non-zero integers from 1-51
        sample = tuple(sorted(random.sample(range(1, 51), 4)))
        
        # Group the first and fourth smallest in one group
        # and the middle two in another
        group1 = (sample[0], sample[3])
        group2 = (sample[1], sample[2])
        
        # Add the sample as a hashable tuple of tuples
        samples.add((group1, group2))
    
    # Convert set to list
    samples_list = list(samples)
    
    # Generate prompts
    prompts = []
    for group1, group2 in samples_list:
        prompt = template.format(a=group1[0], b=group1[1], c=group2[0], d=group2[1])
        prompts.append(prompt)
    
    return prompts

def generate_and_analyze(model_name, llm, tokenizer, prompts, temperature, max_tokens, output_dir, top_n, samples_per_prompt=1):
    """
    Generate responses for prompts and analyze token entropy.
    
    Args:
        model_name: Name of the model
        llm: Initialized LLM instance
        tokenizer: Model tokenizer
        prompts: List of prompts to process
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        output_dir: Directory to save results
        top_n: Number of top tokens to display
        samples_per_prompt: Number of samples to generate per prompt
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompts to file
    prompts_file = os.path.join(output_dir, "comparison_prompts.txt")
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    print(f"Saved {len(prompts)} prompts to {prompts_file}")
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=-1,
        logprobs=50,  # Request logprobs for top 80 tokens
    )
    
    # Process all prompts in a single batch
    token_entropy_data = defaultdict(list)
    token_occurrences = defaultdict(int)
    
    # Add a counter for token frequencies
    token_frequencies = defaultdict(int)
    
    # Store full generations
    generations = []
    
    # Create a mapping of prompts to unique IDs
    prompt_to_id = {prompt: f"prompt_{i}" for i, prompt in enumerate(prompts)}
    
    # Prepare the prompts for multiple samples per prompt
    if samples_per_prompt > 1:
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * samples_per_prompt)
        print(f"Processing {len(prompts)} prompts with {samples_per_prompt} samples each ({len(expanded_prompts)} total generations)")
        batch_prompts = expanded_prompts
    else:
        print(f"Processing all {len(prompts)} prompts in a single batch")
        batch_prompts = prompts
        
    try:
        # Generate with vLLM in a single batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process each output
        for output in outputs:
            generated_text = output.outputs[0].text
            prompt_id = prompt_to_id[output.prompt]
            generations.append({
                "prompt_id": prompt_id,
                "prompt": output.prompt,
                "generated_text": generated_text
            })
            
            if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                generated_token_ids = output.outputs[0].token_ids
                
                # Count token frequencies across all generations
                for token_id in generated_token_ids:
                    token_frequencies[token_id] += 1
                
                for i, token_logprobs in enumerate(output.outputs[0].logprobs):
                    if not token_logprobs:  # Skip if no logprobs for this token
                        continue
                    
                    # Convert logprobs to float values
                    float_logprobs = {}
                    for token_id, logprob in token_logprobs.items():
                        float_logprobs[token_id] = logprob.logprob
                    
                    sorted_logprobs = sorted(float_logprobs.items(), key=lambda x: x[1], reverse=True)
                    if sum(np.exp(logprob) for _, logprob in sorted_logprobs) < 0.99:
                        print(f"Warning: Sum of probabilities is only {sum(np.exp(logprob) for _, logprob in sorted_logprobs):.4f}, less than 0.99")
                    
                    # Get the actual token that was sampled
                    if i < len(generated_token_ids):
                        selected_token_id = generated_token_ids[i]
                        
                        # Calculate entropy
                        logprobs_array = np.array([lp for _, lp in sorted_logprobs])
                        probs = np.exp(logprobs_array)
                        probs = probs / np.sum(probs)  # Normalize
                        entropy = -np.sum(probs * logprobs_array)
                        
                        # Store token and its entropy
                        token_entropy_data[selected_token_id].append(float(entropy))
                        token_occurrences[selected_token_id] += 1
    
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
    
    # Calculate average entropy for each token
    avg_token_entropy = {}
    for token, entropies in token_entropy_data.items():
        avg_token_entropy[token] = sum(entropies) / len(entropies)
    
    # Save generations to disk
    generations_file = os.path.join(output_dir, "generations.json")
    with open(generations_file, 'w', encoding='utf-8') as f:
        json.dump(generations, f, ensure_ascii=False, indent=2)
    
    # Save token entropy data to disk
    entropy_file = os.path.join(output_dir, "token_entropy_data.json")
    with open(entropy_file, 'w', encoding='utf-8') as f:
        json.dump({
            "avg_token_entropy": avg_token_entropy,
            "token_occurrences": token_occurrences
        }, f, ensure_ascii=False, indent=2)
    
    # Save token frequency data to disk
    frequency_file = os.path.join(output_dir, "token_frequency_data.json")
    with open(frequency_file, 'w', encoding='utf-8') as f:
        json.dump({
            "token_frequencies": token_frequencies
        }, f, ensure_ascii=False, indent=2)
    
    # Find top N tokens with highest entropy
    top_tokens = heapq.nlargest(top_n, avg_token_entropy.items(), key=lambda x: x[1])
    
    print(f"\nTop {top_n} tokens with highest average entropy:")
    print("-" * 85)
    print(f"{'Token ID':<10} | {'Token':<25} | {'Avg Entropy':<15} | {'Occurrences':<10}")
    print("-" * 85)
    for token, avg_entropy in top_tokens:
        occurrences = token_occurrences[token]
        # Use repr to make whitespace and special characters visible
        display_token = repr(tokenizer.decode([token]))
        print(f"{token:<10} | {display_token[:25]:<25} | {avg_entropy:<15.4f} | {occurrences:<10}")
    
    # Find top N most frequent tokens
    top_frequent_tokens = heapq.nlargest(top_n, token_frequencies.items(), key=lambda x: x[1])
    
    print(f"\nTop {top_n} most frequent tokens:")
    print("-" * 65)
    print(f"{'Token ID':<10} | {'Token':<25} | {'Frequency':<15}")
    print("-" * 65)
    for token, frequency in top_frequent_tokens:
        # Use repr to make whitespace and special characters visible
        display_token = repr(tokenizer.decode([token]))
        print(f"{token:<10} | {display_token[:25]:<25} | {frequency:<15}")
    
    print(f"\nFull generations saved to {generations_file}")
    print(f"Token entropy data saved to {entropy_file}")
    print(f"Token frequency data saved to {frequency_file}")

if __name__ == "__main__":
    args = parse_args()
    
    # Get templates
    all_templates = get_prompt_templates()
    
    # Filter templates if specified
    if args.template_names:
        template_names = [name.strip() for name in args.template_names.split(',')]
        templates = {name: all_templates[name] for name in template_names if name in all_templates}
        if not templates:
            print(f"Warning: None of the specified template names were found. Using all templates.")
            templates = all_templates
    else:
        templates = all_templates
    
    # Print available templates
    print(f"Available templates:")
    for name in templates.keys():
        print(f"  - {name}")
    
    # Initialize vLLM with the model once
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, max_logprobs=151936)
    tokenizer = llm.get_tokenizer()
    
    # Process each template with a progress bar
    print(f"\nProcessing {len(templates)} templates...")
    for template_name, template in tqdm(templates.items(), desc="Templates", unit="template"):
        print(f"\n{'-'*80}")
        print(f"Processing template: {template_name}")
        print(f"{'-'*80}")
        
        # Create template-specific output directory
        template_output_dir = os.path.join(args.output_dir, template_name)
        os.makedirs(template_output_dir, exist_ok=True)
        
        print(f"Generating {args.num_samples} comparison samples with random seed {args.random_seed}")
        prompts = generate_comparison_samples(template, args.num_samples, args.random_seed)
        
        generate_and_analyze(
            args.model,
            llm,
            tokenizer,
            prompts,
            args.temperature,
            args.max_tokens,
            template_output_dir,
            args.top_n,
            args.samples_per_prompt
        )