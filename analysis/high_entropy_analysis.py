import json
import argparse
import os
from tabulate import tabulate
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token entropy data")
    parser.add_argument("--input_dir", type=str, default="entropy_stats_output", 
                        help="Directory containing token entropy data")
    parser.add_argument("--min_entropy", type=float, default=0.5,
                        help="Minimum entropy threshold for filtering")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top tokens to display")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model to use for tokenization")
    return parser.parse_args()

def analyze_token_entropy(input_dir, min_entropy, top_n, model_name):
    # Load token entropy data
    entropy_file = os.path.join(input_dir, "token_entropy_data.json")
    
    if not os.path.exists(entropy_file):
        print(f"Error: Token entropy data file not found at {entropy_file}")
        return
    
    with open(entropy_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    avg_token_entropy = data.get("avg_token_entropy", {})
    token_occurrences = data.get("token_occurrences", {})
    
    # Convert string keys to integers (JSON serializes dict keys as strings)
    avg_token_entropy = {int(k): v for k, v in avg_token_entropy.items()}
    token_occurrences = {int(k): v for k, v in token_occurrences.items()}
    
    # Filter tokens with entropy > min_entropy
    high_entropy_tokens = {token: entropy for token, entropy in avg_token_entropy.items() 
                          if entropy > min_entropy}
    
    if not high_entropy_tokens:
        print(f"No tokens found with entropy > {min_entropy}")
        return
    
    # Get occurrences for filtered tokens
    filtered_tokens = [(token, entropy, token_occurrences.get(token, 0)) 
                       for token, entropy in high_entropy_tokens.items()]
    
    # Sort by occurrences (descending)
    sorted_tokens = sorted(filtered_tokens, key=lambda x: x[2], reverse=True)
    
    # Limit to top_n
    top_tokens = sorted_tokens[:top_n]
    
    # Load the model tokenizer
    try:
    
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Prepare table data with decoded tokens
        table_data = []
        for token_id, entropy, occurrences in top_tokens:
            try:
                token_text = repr(tokenizer.decode([token_id]))
                table_data.append([token_id, token_text, entropy, occurrences])
            except Exception as e:
                # Fallback if decoding fails
                table_data.append([token_id, f"<decoding_error: {str(e)}>", entropy, occurrences])
    
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        table_data = [[token_id, "<no_decoder>", entropy, occurrences] 
                     for token_id, entropy, occurrences in top_tokens]
    
    # Display results
    headers = ["Token ID", "Token", "Avg Entropy", "Occurrences"]
    print(f"\nTop {len(table_data)} tokens with entropy > {min_entropy}, ranked by occurrences:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    output_file = os.path.join(input_dir, f"high_entropy_tokens_min{min_entropy}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "min_entropy": min_entropy,
            "tokens": [{"id": t[0], "entropy": t[1], "occurrences": t[2]} for t in sorted_tokens]
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    analyze_token_entropy(args.input_dir, args.min_entropy, args.top_n, args.model)