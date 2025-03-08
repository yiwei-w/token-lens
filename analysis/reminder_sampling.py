#!/usr/bin/env python
# backend/reminder_sampling.py
import json
import random
import re
from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
import time

random.seed(42)

# Reminder variations to insert
REMINDERS = [
    "Okay, I need to remember not to use numerical approximations.",
    "Well, I should avoid converting to decimals.",
    "Wait, let me ensure I'm not using numerical approximations."
]

def load_input_data():
    """Load and deduplicate input data from both JSON files."""
    add_path = "./entropy_stats_output_all_templates/add_no_approx/generations.json"
    sub_path = "./entropy_stats_output_all_templates/sub_no_approx/generations.json"
    
    with open(add_path, 'r') as f:
        add_data = json.load(f)
    
    with open(sub_path, 'r') as f:
        sub_data = json.load(f)
    
    # Deduplicate by prompt_id
    add_dict = {item["prompt_id"]: item for item in add_data}
    sub_dict = {item["prompt_id"]: item for item in sub_data}
    
    # Verify we have 300 unique prompts in each
    print(f"Unique add prompts: {len(add_dict)}")
    print(f"Unique sub prompts: {len(sub_dict)}")
    assert len(add_dict) == 300
    assert len(sub_dict) == 300
    
    # Combine them
    return list(add_dict.values()), list(sub_dict.values())

def generate_with_repeated_reminders(prompt, llm, max_total_tokens=2048, max_reminders=3):
    """Generate text with reminders after paragraph breaks, repeating the process."""
    print("Starting generation with repeated reminders...")
    
    # Token counter for the entire generation (approximate)
    total_tokens = 0
    reminder_count = 0
    
    # The text we've generated so far
    full_generated_text = ""
    
    # Current working prompt
    current_prompt = prompt
    
    while reminder_count < max_reminders and total_tokens < max_total_tokens - 200:
        # Parameters for next segment generation
        segment_max_tokens = min(512, max_total_tokens - total_tokens)
        if segment_max_tokens <= 20:  # Too few tokens left
            break
            
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=segment_max_tokens,
        )
        
        # Generate next segment
        outputs = llm.generate([current_prompt], sampling_params)
        segment_text = outputs[0].outputs[0].text
        
        # Check if we have a paragraph break that's not ":\n\n"
        has_break = False
        break_pos = -1
        
        # Find the first occurrence of \n\n, ensuring it's not :\n\n
        for match in re.finditer(r'\n\n', segment_text):
            pos = match.start()
            if pos > 0 and segment_text[pos-1] != ':':
                has_break = True
                break_pos = pos
                break  # Stop at the first proper paragraph break
        
        if not has_break:
            # No more paragraph breaks found, just append what we have and stop
            full_generated_text += segment_text
            # Approximate token count
            total_tokens += len(segment_text.split())
            break
        
        # Keep text up to the break (including the \n\n)
        kept_text = segment_text[:break_pos+2]
        full_generated_text += kept_text
        
        # Choose a random reminder
        reminder = random.choice(REMINDERS)
        print(f"Inserted reminder {reminder_count+1}: {reminder}")
        
        # Add reminder to our output
        full_generated_text += reminder
        reminder_count += 1
        
        # Update current prompt for next segment
        current_prompt = prompt + full_generated_text
        
        # Update approximate token count
        total_tokens += len(kept_text.split()) + len(reminder.split())
    
    # If we have tokens left and didn't hit max reminders, generate one final segment
    remaining_tokens = max_total_tokens - total_tokens
    if remaining_tokens > 100:
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=remaining_tokens,
        )
        
        outputs = llm.generate([current_prompt], sampling_params)
        final_segment = outputs[0].outputs[0].text
        
        full_generated_text += final_segment
    
    return full_generated_text

def generate_with_reminders(prompts, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Generate completions with repeated reminder insertions at paragraph breaks."""
    print(f"Initializing LLM: {model_name}")
    
    # Initialize the model
    llm = LLM(model=model_name, trust_remote_code=True)
    
    # Prepare prompts for generation
    prompt_texts = [item["prompt"] for item in prompts]
    prompt_ids = [item["prompt_id"] for item in prompts]
    
    # Generate outputs
    outputs = []
    
    # Process one by one to ensure proper reminder insertion
    for i in tqdm(range(len(prompt_texts)), desc="Generating"):
        prompt = prompt_texts[i]
        prompt_id = prompt_ids[i]
        
        try:
            # Generate with multiple reminders inserted at paragraph breaks
            generated_text = generate_with_repeated_reminders(
                prompt, 
                llm, 
                max_total_tokens=2048,  # Max total tokens
                max_reminders=3         # Max number of reminders to insert
            )
            
            # Save the generation
            outputs.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generated_text": generated_text
            })
            
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
        
        # # Small delay to prevent potential issues
        # time.sleep(0.1)
    
    return outputs

def main():
    # Load input data
    print("Loading and deduplicating input data...")
    add_data, sub_data = load_input_data()
    
    # Generate completions with reminders
    print("Generating completions with reminders for addition problems...")
    add_outputs = generate_with_reminders(add_data)
    
    print("Generating completions with reminders for subtraction problems...")
    sub_outputs = generate_with_reminders(sub_data)
    
    # Save results to separate files
    add_output_path = "./reminder_generations_add.json"
    sub_output_path = "./reminder_generations_sub.json"
    
    print(f"Saving {len(add_outputs)} addition generations to {add_output_path}")
    with open(add_output_path, 'w') as f:
        json.dump(add_outputs, f, indent=2)
    
    print(f"Saving {len(sub_outputs)} subtraction generations to {sub_output_path}")
    with open(sub_output_path, 'w') as f:
        json.dump(sub_outputs, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()