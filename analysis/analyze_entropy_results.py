import os
import json
from collections import defaultdict
import heapq
from tabulate import tabulate
import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token entropy data from token_entropy_stats output")
    parser.add_argument("--data_dir", type=str, default="entropy_stats_output_all_templates",
                        help="Directory containing the entropy stats output")
    parser.add_argument("--top_n", type=int, default=100,
                        help="Number of top entropy tokens to display")
    parser.add_argument("--min_entropy", type=float, default=0.5,
                        help="Minimum entropy threshold for frequency analysis")
    return parser.parse_args()


def load_tokenizer():
    """Load the DeepSeek tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Continuing with token IDs only...")
        return None


def get_token_display(token_id, tokenizer):
    """Get a display representation for a token ID using the tokenizer"""
    if tokenizer is not None:
        try:
            # Convert token ID to text
            token_text = tokenizer.decode([token_id])
            # Clean up the representation for display
            if token_text.strip() == "":
                # For whitespace tokens, show a more readable representation
                token_bytes = token_text.encode('unicode_escape')
                return f"'{token_bytes.decode('utf-8')}'"
            return f"'{token_text}'"
        except Exception:
            pass
    return f"Token_{token_id}"


def analyze_template_data(template_dir, tokenizer, top_n=20, min_entropy=0.5):
    """Analyze entropy and frequency data for a single template"""
    entropy_file = os.path.join(template_dir, "token_entropy_data.json")
    frequency_file = os.path.join(template_dir, "token_frequency_data.json")
    
    if not os.path.exists(entropy_file) or not os.path.exists(frequency_file):
        print(f"Missing data files in {template_dir}")
        return None, None
    
    # Load entropy data
    with open(entropy_file, 'r', encoding='utf-8') as f:
        entropy_data = json.load(f)
    
    # Load frequency data
    with open(frequency_file, 'r', encoding='utf-8') as f:
        frequency_data = json.load(f)
    
    avg_token_entropy = entropy_data["avg_token_entropy"]
    token_occurrences = entropy_data["token_occurrences"]
    token_frequencies = frequency_data["token_frequencies"]
    
    # Convert string keys to integers for consistent sorting
    avg_token_entropy = {int(k): v for k, v in avg_token_entropy.items()}
    token_occurrences = {int(k): v for k, v in token_occurrences.items()}
    token_frequencies = {int(k): v for k, v in token_frequencies.items()}
    
    # Find top N tokens with highest entropy
    top_entropy_tokens = heapq.nlargest(top_n, avg_token_entropy.items(), key=lambda x: x[1])
    
    # Create a list of high-entropy tokens with their frequencies
    high_entropy_tokens = []
    for token_id, entropy in avg_token_entropy.items():
        if entropy >= min_entropy:
            frequency = token_frequencies.get(token_id, 0)
            high_entropy_tokens.append((token_id, entropy, frequency))
    
    # Sort by frequency (descending)
    high_freq_high_entropy = sorted(high_entropy_tokens, key=lambda x: x[2], reverse=True)[:top_n]
    
    return top_entropy_tokens, high_freq_high_entropy


def main():
    args = parse_args()
    
    # Load tokenizer instead of token map
    tokenizer = load_tokenizer()
    
    # Get all template directories
    template_dirs = []
    for item in os.listdir(args.data_dir):
        item_path = os.path.join(args.data_dir, item)
        if os.path.isdir(item_path):
            template_dirs.append(item_path)
    
    if not template_dirs:
        print(f"No template directories found in {args.data_dir}")
        return
    
    print(f"Found {len(template_dirs)} template directories")
    
    # Analyze each template
    all_high_entropy_tokens = defaultdict(list)
    all_high_freq_tokens = defaultdict(list)
    
    for template_dir in template_dirs:
        template_name = os.path.basename(template_dir)
        print(f"\nAnalyzing template: {template_name}")
        
        top_entropy, high_freq = analyze_template_data(
            template_dir, 
            tokenizer,  # Pass tokenizer instead of token_map
            args.top_n, 
            args.min_entropy
        )
        
        if top_entropy and high_freq:
            all_high_entropy_tokens[template_name] = top_entropy
            all_high_freq_tokens[template_name] = high_freq
            
            # Display top entropy tokens for this template
            print(f"\nTop {args.top_n} tokens with highest entropy for {template_name}:")
            headers = ["Token ID", "Token", "Avg Entropy"]
            table_data = [
                [token_id, get_token_display(token_id, tokenizer), f"{entropy:.4f}"]
                for token_id, entropy in top_entropy
            ]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Display high frequency, high entropy tokens for this template
            print(f"\nTop {args.top_n} most frequent tokens with entropy > {args.min_entropy} for {template_name}:")
            headers = ["Token ID", "Token", "Avg Entropy", "Frequency"]
            table_data = [
                [token_id, get_token_display(token_id, tokenizer), f"{entropy:.4f}", frequency]
                for token_id, entropy, frequency in high_freq
            ]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Cross-template analysis
    print("\n" + "="*80)
    print("Cross-Template Analysis")
    print("="*80)
    
    # Collect tokens that appear in high entropy lists across multiple templates
    token_template_count = defaultdict(int)
    token_entropy_by_template = defaultdict(dict)
    
    for template_name, tokens in all_high_entropy_tokens.items():
        for token_id, entropy in tokens:
            token_template_count[token_id] += 1
            token_entropy_by_template[token_id][template_name] = entropy
    
    # Find tokens that appear in multiple templates
    common_tokens = [(token_id, count) for token_id, count in token_template_count.items() if count > 1]
    common_tokens.sort(key=lambda x: x[1], reverse=True)
    
    if common_tokens:
        print("\nTokens appearing in multiple templates (sorted by number of templates):")
        headers = ["Token ID", "Token", "# Templates", "Avg Entropy (across templates)"]
        table_data = []
        
        for token_id, count in common_tokens[:args.top_n]:
            # Calculate average entropy across templates
            avg_entropy = sum(token_entropy_by_template[token_id].values()) / count
            table_data.append([
                token_id, 
                get_token_display(token_id, tokenizer), 
                count, 
                f"{avg_entropy:.4f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Create a consolidated table of high entropy, high frequency tokens
    all_tokens = {}
    for template_name, tokens in all_high_freq_tokens.items():
        for token_id, entropy, frequency in tokens:
            if token_id not in all_tokens:
                all_tokens[token_id] = {
                    "token_id": token_id,
                    "entropy": entropy,
                    "frequency": frequency,
                    "templates": [template_name]
                }
            else:
                all_tokens[token_id]["templates"].append(template_name)
                all_tokens[token_id]["frequency"] += frequency
                # Update entropy with max value
                all_tokens[token_id]["entropy"] = max(all_tokens[token_id]["entropy"], entropy)
    
    # Convert to list and sort by frequency
    consolidated_tokens = list(all_tokens.values())
    consolidated_tokens.sort(key=lambda x: x["frequency"], reverse=True)
    
    print("\nConsolidated table of high entropy (>{:.1f}), high frequency tokens across all templates:"
          .format(args.min_entropy))
    headers = ["Token ID", "Token", "Max Entropy", "Total Frequency", "# Templates"]
    table_data = [
        [
            t["token_id"], 
            get_token_display(t["token_id"], tokenizer), 
            f"{t['entropy']:.4f}", 
            t["frequency"], 
            len(t["templates"])
        ]
        for t in consolidated_tokens[:args.top_n]
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main() 