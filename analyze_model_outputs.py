import pandas as pd
import os
from os import walk
from os.path import basename
from collections import defaultdict
from nltk.tokenize import word_tokenize

def get_file_groups_by_dir(root_dir):
    file_groups = defaultdict(lambda: {'prompts': None, 'outputs': []})
    
    # Find all CSV files
    for root, _, files in walk(root_dir):
        # Convert Windows path separators to "/"
        root = root.replace("\\", "/")
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            continue
            
        for file in csv_files:
            full_path = f"{root}/{file}"  # Use "/" directly for path creation
            if file == 'prompts.csv':
                file_groups[root]['prompts'] = full_path
            else:
                file_groups[root]['outputs'].append(full_path)
                
    # Remove groups without prompts.csv
    return {k: v for k, v in file_groups.items() if v['prompts'] and v['outputs']}

def add_length_metrics_to_outputs(file_groups):
    results_dict = {}
    
    for folder, files in file_groups.items():
        print(f"\nProcessing folder: {folder}")
        folder_results = {}

        # Load prompts.csv
        prompts_df = pd.read_csv(files['prompts'])
        prompts_df['target_length'] = prompts_df['text'].apply(lambda x: len(word_tokenize(str(x))))

        # Process each model output file
        for output_file in files['outputs']:
            print(f"Processing: {basename(output_file)}")
            model_results = {}

            try:
                # Load model output with error handling
                try:
                    model_df = pd.read_csv(output_file)
                except pd.errors.ParserError as e:
                    print(f"Error reading {basename(output_file)}: {str(e)}")
                    print(f"Skipping {basename(output_file)} due to parsing error")
                    continue

                # Check if file already has length metrics
                if 'generated_length' in model_df.columns and 'target_length' in model_df.columns:
                    print(f"- File already has length metrics, skipping...")
                    model_results['dataframe'] = model_df
                    model_results['wrong_count'] = len(model_df[model_df['writing'] == "SOMETHING_WRONG"])
                    model_results['valid_count'] = len(model_df) - model_results['wrong_count']
                else:
                    # Replace NA values with "SOMETHING_WRONG"
                    model_df['writing'] = model_df['writing'].fillna("SOMETHING_WRONG")

                    # Count SOMETHING_WRONG rows
                    wrong_count = len(model_df[model_df['writing'] == "SOMETHING_WRONG"])
                    print(f"- Number of 'SOMETHING_WRONG' found: {wrong_count}")

                    # Calculate word count, setting NA for SOMETHING_WRONG rows
                    model_df['generated_length'] = model_df.apply(
                        lambda row: len(word_tokenize(str(row['writing']))) 
                        if row['writing'] != "SOMETHING_WRONG" 
                        else pd.NA, 
                        axis=1
                    )
                    model_df['target_length'] = prompts_df['target_length']

                    # Save the updated DataFrame back to CSV
                    model_df.to_csv(output_file, index=False)
                    print(f"- Updated file saved: {basename(output_file)}")
                    
                    model_results['dataframe'] = model_df
                    model_results['wrong_count'] = wrong_count
                    model_results['valid_count'] = len(model_df) - wrong_count

            except Exception as e:
                print(f"Error processing {basename(output_file)}: {str(e)}")
                continue

            model_name = basename(output_file).replace('.csv', '')
            folder_results[model_name] = model_results

        results_dict[folder] = folder_results
    
    return results_dict

def calculate_summary_statistics(results_dict):
    # Initialize list to store statistics table
    summary_stats = []

    for folder, folder_results in results_dict.items():
        # Extract setting and dataset from folder name
        setting = folder.split('/')[1]  # Setting1, Setting4, ...
        dataset = folder.split('/')[2]  # blog, CCAT50, enron, reddit
        
        for model_name, model_results in folder_results.items():
            df = model_results['dataframe']
            wrong_count = model_results['wrong_count']
            valid_count = model_results['valid_count']
            
            # Calculate statistics for valid samples
            valid_df = df[df['writing'] != "SOMETHING_WRONG"]
            
            length_diff = valid_df['generated_length'] - valid_df['target_length']
            length_diff_mean = round(length_diff.mean(), 2)
            length_diff_std = round(length_diff.std(), 2)

            # Calculate statistics
            stats = {
                'Setting': setting,
                'Dataset': dataset,
                'Model': model_name,
                'Total Samples': len(df),
                'Valid Samples': valid_count,
                'Invalid Samples': wrong_count,
                'Valid Percentage': round((valid_count / len(df)) * 100, 2),
                'Avg Generated Length': round(valid_df['generated_length'].mean(), 2),
                'Std Generated Length': round(valid_df['generated_length'].std(), 2),
                'Min Generated Length': valid_df['generated_length'].min(),
                'Max Generated Length': valid_df['generated_length'].max(),
                'Avg Target Length': round(valid_df['target_length'].mean(), 2),
                'Std Target Length': round(valid_df['target_length'].std(), 2),
                'Length Difference': f"{length_diff_mean} ± {length_diff_std}"
            }
            summary_stats.append(stats)

    # Create statistics table
    summary_df = pd.DataFrame(summary_stats)

    # Sort column order
    columns = ['Setting', 'Dataset', 'Model', 'Total Samples', 'Valid Samples', 
               'Invalid Samples', 'Valid Percentage', 'Avg Generated Length', 
               'Std Generated Length', 'Min Generated Length', 'Max Generated Length',
               'Avg Target Length', 'Std Target Length', 'Length Difference']
    summary_df = summary_df[columns]

    return summary_df

def main():
    # Set root directory
    root_directory = "LLM_writing"  # Jack: changed from "LLM_Writing" to "LLM_writing"
    
    # Get file groups
    file_groups = get_file_groups_by_dir(root_directory)
    
    # Process files and get results
    results_dict = add_length_metrics_to_outputs(file_groups)
    
    # Calculate and display summary statistics
    summary_df = calculate_summary_statistics(results_dict)
    print("\nSummary Statistics Table:")
    print(summary_df)
    
    # Create results directory and save summary statistics
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary statistics to CSV in results directory
    output_path = os.path.join(results_dir, "model_output_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary statistics saved to: {output_path}")

if __name__ == "__main__":
    main() 