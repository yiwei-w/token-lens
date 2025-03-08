import json
import argparse
import os
import re
from tabulate import tabulate
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze tokens starting with capital letters")
    parser.add_argument("--input_dir", type=str, default="entropy_stats_output", 
                        help="Directory containing token entropy data")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top tokens to display")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model to use for tokenization")
    return parser.parse_args()

def analyze_capital_tokens(input_dir, top_n, model_name):
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
    
    # Load the model tokenizer
    try:
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Find tokens that start with capital letters
        capital_tokens = []
        for token_id, entropy in avg_token_entropy.items():
            try:
                token_text = tokenizer.decode([token_id])
                # Strip non-alphabetic characters and check if it starts with uppercase
                stripped_text = re.sub(r'[^a-zA-Z]', '', token_text)
                if stripped_text and stripped_text[0].isupper():
                    occurrences = token_occurrences.get(token_id, 0)
                    capital_tokens.append((token_id, token_text, entropy, occurrences))
            except Exception as e:
                print(f"Error decoding token {token_id}: {str(e)}")
                continue
        
        if not capital_tokens:
            print("No tokens found that start with capital letters")
            return
        
        # Sort by entropy (descending)
        sorted_tokens = sorted(capital_tokens, key=lambda x: x[2], reverse=True)
        
        # Limit to top_n
        top_tokens = sorted_tokens[:top_n]
        
        # Prepare table data
        table_data = []
        for token_id, token_text, entropy, occurrences in top_tokens:
            table_data.append([token_id, repr(token_text), entropy, occurrences])
    
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return
    
    # Display results
    headers = ["Token ID", "Token", "Avg Entropy", "Occurrences"]
    print(f"\nTop {len(table_data)} tokens starting with capital letters, ranked by entropy:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    output_file = os.path.join(input_dir, "capital_letter_tokens.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "tokens": [{"id": t[0], "text": t[1], "entropy": t[2], "occurrences": t[3]} 
                      for t in sorted_tokens]
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    analyze_capital_tokens(args.input_dir, args.top_n, args.model)