#!/usr/bin/env python
# backend/analyze_reminders.py
import json
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reminder patterns to exclude from analysis
REMINDER_PATTERNS = [
    r"Wait, I need to remember not to use numerical approximations here",
    r"I should avoid converting to decimals and work with exact radical expressions",
    r"Let me ensure I'm working with symbolic expressions only, not approximations"
]

def load_data():
    """Load original and reminder-augmented generations."""
    # Load original data
    add_original_path = "./entropy_stats_output_all_templates/add_no_approx/generations.json"
    sub_original_path = "./entropy_stats_output_all_templates/sub_no_approx/generations.json"
    
    with open(add_original_path, 'r') as f:
        add_original = json.load(f)
    
    with open(sub_original_path, 'r') as f:
        sub_original = json.load(f)
    
    # Load reminder-augmented data from separate files
    add_reminder_path = "./reminder_generations_add.json"
    sub_reminder_path = "./reminder_generations_sub.json"
    
    with open(add_reminder_path, 'r') as f:
        add_reminder_data = json.load(f)
    
    with open(sub_reminder_path, 'r') as f:
        sub_reminder_data = json.load(f)
    
    # Create dictionaries for original data
    add_original_dict = {item["prompt_id"]: item for item in add_original}
    sub_original_dict = {item["prompt_id"]: item for item in sub_original}
    
    # Create dictionaries for reminder data
    add_reminder_dict = {item["prompt_id"]: item for item in add_reminder_data}
    sub_reminder_dict = {item["prompt_id"]: item for item in sub_reminder_data}
    
    return {
        "add_original": add_original_dict,
        "sub_original": sub_original_dict,
        "add_reminder": add_reminder_dict,
        "sub_reminder": sub_reminder_dict
    }

def has_approximation(text, exclude_reminders=True):
    """Check if text contains approximation indicators."""
    # Create a pattern to exclude the reminder text itself
    exclude_pattern = None
    if exclude_reminders:
        exclude_pattern = "|".join(REMINDER_PATTERNS)
    
    # Define approximation patterns
    approx_patterns = [
        r'approximately', 
        r'≈',
        r'~=',
        # Look for decimal points in numbers (but not when part of reminder)
        r'\b\d+\.\d+\b'
    ]
    
    for pattern in approx_patterns:
        if exclude_reminders and exclude_pattern:
            # Skip matches that are part of a reminder
            for match in re.finditer(pattern, text, re.IGNORECASE):
                match_pos = match.start()
                match_text = text[match_pos:match_pos+100]  # Extract some context
                
                # Check if match is within a reminder
                if not any(re.search(reminder_pat, match_text, re.IGNORECASE) for reminder_pat in REMINDER_PATTERNS):
                    return True
        else:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    
    return False

def find_first_approx_position(text):
    """Find the token index where approximation first appears."""
    # Convert text to approximate tokens
    # This is a simple approximation of tokenization
    tokens = text.split()
    
    for i, token in enumerate(tokens):
        # Check if token contains a decimal or approximation symbol
        if re.search(r'\d+\.\d+', token) or '≈' in token or '~=' in token:
            return i
        
        # Check for 'approximately'
        if 'approximately' in token:
            return i
    
    return -1  # Not found

def analyze_outputs(data_dict):
    """Analyze outputs to check for approximations."""
    results = {}
    
    for data_type, data in data_dict.items():
        print(f"Analyzing {data_type}...")
        
        # Counters
        total = len(data)
        with_approx = 0
        approx_positions = []
        
        for item_id, item in tqdm(data.items()):
            text = item["generated_text"]
            
            # Check if the completion contains approximations
            if has_approximation(text):
                with_approx += 1
                
                # Find where the approximation first appears
                pos = find_first_approx_position(text)
                if pos >= 0:
                    approx_positions.append(pos)
        
        # Calculate percentage
        approx_percentage = (with_approx / total) * 100 if total > 0 else 0
        
        # Calculate average position
        avg_pos = np.mean(approx_positions) if approx_positions else -1
        
        results[data_type] = {
            "total": total,
            "with_approx": with_approx,
            "approx_percentage": approx_percentage,
            "avg_first_approx_position": avg_pos
        }
    
    return results

def visualize_results(results):
    """Create visualizations for the analysis results."""
    # Create data for plotting
    labels = []
    orig_percentages = []
    reminder_percentages = []
    
    for op in ['add', 'sub']:
        labels.append(op)
        orig_percentages.append(results[f"{op}_original"]["approx_percentage"])
        reminder_percentages.append(results[f"{op}_reminder"]["approx_percentage"])
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot percentages
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, orig_percentages, width, label='Original')
    ax1.bar(x + width/2, reminder_percentages, width, label='With Reminders')
    
    ax1.set_ylabel('Percentage with Approximations')
    ax1.set_title('Impact of Reminders on Approximation Usage')
    ax1.set_xticks(x)
    ax1.set_xticklabels([label+"_no_approx" for label in labels])
    ax1.legend()
    
    # Plot average position
    positions_orig = [results[f"{op}_original"]["avg_first_approx_position"] for op in ['add', 'sub']]
    positions_reminder = [results[f"{op}_reminder"]["avg_first_approx_position"] for op in ['add', 'sub']]
    
    ax2.bar(x - width/2, positions_orig, width, label='Original')
    ax2.bar(x + width/2, positions_reminder, width, label='With Reminders')
    
    ax2.set_ylabel('Avg Token Position of First Approximation')
    ax2.set_title('When Approximations First Appear')
    ax2.set_xticks(x)
    ax2.set_xticklabels([label+"_no_approx" for label in labels])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("./reminder_analysis_results.png", dpi=300)
    plt.close()

def main():
    # Load data
    print("Loading data...")
    data_dict = load_data()
    
    # Analyze outputs
    print("Analyzing outputs...")
    results = analyze_outputs(data_dict)
    
    # Display results
    print("\nAnalysis Results:")
    for data_type, stats in results.items():
        print(f"\n{data_type}:")
        print(f"  Total completions: {stats['total']}")
        print(f"  Completions with approximations: {stats['with_approx']} ({stats['approx_percentage']:.2f}%)")
        print(f"  Average position of first approximation: {stats['avg_first_approx_position']:.2f}")
    
    # Create overall comparison
    print("\nOverall Comparison:")
    add_diff = results["add_original"]["approx_percentage"] - results["add_reminder"]["approx_percentage"]
    sub_diff = results["sub_original"]["approx_percentage"] - results["sub_reminder"]["approx_percentage"]
    
    print(f"  Addition problems: Reminders reduced approximations by {add_diff:.2f}%")
    print(f"  Subtraction problems: Reminders reduced approximations by {sub_diff:.2f}%")
    
    # Visualize results
    print("\nCreating visualizations...")
    visualize_results(results)
    
    # Save results to JSON
    print("Saving results...")
    with open("./reminder_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()