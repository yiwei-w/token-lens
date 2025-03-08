import os
import json
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate histograms of token entropy distributions")
    parser.add_argument("--data_dir", type=str, default="entropy_stats_output_all_templates",
                        help="Directory containing the entropy stats output")
    parser.add_argument("--bins", type=int, default=20,
                        help="Number of bins for the histogram")
    parser.add_argument("--output_dir", type=str, default="entropy_histograms",
                        help="Directory to save histogram images")
    parser.add_argument("--min_frequency", type=int, default=5,
                        help="Minimum token frequency to include in analysis")
    return parser.parse_args()


def load_template_data(template_dir):
    """Load entropy and frequency data for a single template"""
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
    
    return entropy_data, frequency_data


def create_weighted_histogram(entropy_values, frequencies, bins=20, title="Token Entropy Distribution", 
                              filename=None):
    """Create and save a histogram of entropy values weighted by token frequency"""
    plt.figure(figsize=(12, 7))
    
    # Weighted histogram (by token frequency)
    plt.hist(entropy_values, bins=bins, weights=frequencies, alpha=0.7, color='purple', 
             edgecolor='black')
    plt.title(title)
    plt.xlabel("Entropy")
    plt.ylabel("Total Token Occurrences")
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {filename}")
    
    plt.close()


def create_scatter_plot(entropy_values, frequencies, title="Entropy vs Frequency", 
                        filename=None, log_scale=False):
    """Create a scatter plot of entropy vs frequency"""
    plt.figure(figsize=(12, 7))
    plt.scatter(entropy_values, frequencies, alpha=0.5)
    plt.title(title)
    plt.xlabel("Entropy")
    
    if log_scale:
        plt.ylabel("Frequency (log scale)")
        plt.yscale('log')
    else:
        plt.ylabel("Frequency")
    
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {filename}")
    
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
    
    # For aggregated data across all templates
    all_entropy_values = []
    all_frequencies = []
    
    # For grid of histograms
    template_data = []
    
    # Process each template
    for template_dir in template_dirs:
        template_name = os.path.basename(template_dir)
        print(f"Processing template: {template_name}")
        
        entropy_data, frequency_data = load_template_data(template_dir)
        
        if not entropy_data or not frequency_data:
            continue
        
        # Extract data
        avg_token_entropy = {int(k): v for k, v in entropy_data["avg_token_entropy"].items()}
        token_frequencies = {int(k): v for k, v in frequency_data["token_frequencies"].items()}
        
        # Filter tokens by minimum frequency
        entropy_values = []
        frequencies = []
        
        for token_id, entropy in avg_token_entropy.items():
            frequency = token_frequencies.get(token_id, 0)
            if frequency >= args.min_frequency:
                entropy_values.append(entropy)
                frequencies.append(frequency)
        
        if not entropy_values:
            print(f"No tokens with frequency >= {args.min_frequency} found for {template_name}")
            continue
        
        # Add to aggregated data
        all_entropy_values.extend(entropy_values)
        all_frequencies.extend(frequencies)
        
        # Store data for grid plot
        template_data.append({
            'name': template_name,
            'entropy_values': entropy_values,
            'frequencies': frequencies
        })
        
        # Create individual histogram for this template
        title = f"Token Entropy Distribution (Weighted by Frequency) - {template_name}"
        filename = os.path.join(args.output_dir, f"{template_name}_histogram.png")
        create_weighted_histogram(entropy_values, frequencies, bins=args.bins, 
                                  title=title, filename=filename)
    
    # Create aggregated histogram across all templates
    if all_entropy_values:
        title = "Token Entropy Distribution (Weighted by Frequency) - All Templates"
        filename = os.path.join(args.output_dir, "all_templates_histogram.png")
        create_weighted_histogram(all_entropy_values, all_frequencies, bins=args.bins, 
                                  title=title, filename=filename)
    
    # Create grid of histograms
    if template_data:
        create_histogram_grid(template_data, args.bins, args.output_dir)


def create_histogram_grid(template_data, bins, output_dir):
    """Create a grid of histograms for all templates"""
    n_templates = len(template_data)
    
    # Calculate grid dimensions (trying to make it as square as possible)
    grid_cols = int(n_templates ** 0.5)
    grid_rows = (n_templates + grid_cols - 1) // grid_cols  # Ceiling division
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
    fig.suptitle("Token Entropy Distributions Across Templates", fontsize=16)
    
    # Flatten axes array for easier indexing if it's 2D
    if grid_rows > 1 and grid_cols > 1:
        axes = axes.flatten()
    elif grid_rows == 1:
        axes = [axes] if grid_cols == 1 else axes
    elif grid_cols == 1:
        axes = [axes] if grid_rows == 1 else axes
    
    # Find global min and max entropy values for unified x-axis
    all_entropy = []
    for template in template_data:
        all_entropy.extend(template['entropy_values'])
    
    if all_entropy:
        x_min = min(all_entropy)
        x_max = max(all_entropy)
    else:
        x_min, x_max = 0, 1  # Default if no data
    
    for i, template in enumerate(template_data):
        if i < len(axes):
            ax = axes[i]
            
            # Weighted histogram
            ax.hist(template['entropy_values'], bins=bins, weights=template['frequencies'], 
                   alpha=0.7, color='purple', edgecolor='black')
            
            ax.set_title(template['name'], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set unified x-axis limits
            ax.set_xlim(x_min, x_max)
            
            # Only add x and y labels for the bottom and left plots
            if i >= len(axes) - grid_cols:  # Bottom row
                ax.set_xlabel("Entropy")
            if i % grid_cols == 0:  # Left column
                ax.set_ylabel("Token Occurrences")
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save the grid figure
    grid_filename = os.path.join(output_dir, "template_histograms_grid.png")
    plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
    print(f"Saved histogram grid to {grid_filename}")
    plt.close()


if __name__ == "__main__":
    main()