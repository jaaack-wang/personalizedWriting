import pandas as pd
import os
from os import walk
from os.path import join
from collections import defaultdict
import ast

def get_test_files():
    test_files = {}
    prepare_dir = "dataset_prepare"
    
    for file in os.listdir(prepare_dir):
        if file.endswith("_test.csv"):
            dataset_type = file.replace("_test.csv", "")
            test_files[dataset_type] = join(prepare_dir, file)
    
    return test_files

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

def calculate_top_k_accuracy(ground_truth, predictions, k):
    if isinstance(predictions, str):
        predictions = ast.literal_eval(predictions)
    
    top_k_preds = predictions[:k]
    return ground_truth in top_k_preds

def compute_accuracies(test_df, pred_df, model_name):
    results = {}
    k_values = [1, 3, 5, 10]
    
    for k in k_values:
        correct = 0
        total = len(test_df)
        
        for i in range(total):
            ground_truth = test_df.iloc[i]['AA-label']
            predictions = pred_df.iloc[i][f'{model_name}-AA-top_k-predictions']
            
            if calculate_top_k_accuracy(ground_truth, predictions, k):
                correct += 1
        
        accuracy = (correct / total) * 100
        results[f'top_{k}'] = round(accuracy, 2)
    
    return results

def main():
    test_files = get_test_files()
    file_groups = get_file_groups_by_dir("LLM_writing")

    # Nested dictionary to store results
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # First calculate results for original test sets
    print("\nCalculating results for original test sets...")
    for dataset, test_file in test_files.items():
        print(f"\nProcessing original test set for {dataset}")
        test_df = pd.read_csv(test_file)
        
        aa_models_test = [col.split('-AA-top_k-predictions')[0] 
                         for col in test_df.columns 
                         if '-AA-top_k-predictions' in col]
        
        if aa_models_test:
            print(f"Found AA models in test set: {', '.join(aa_models_test)}")
            for aa_model in aa_models_test:
                test_results = compute_accuracies(test_df, test_df, aa_model)
                all_results['original'][dataset][aa_model] = test_results
                values = [test_results[f'top_{k}'] for k in [1, 3, 5, 10]]
                print(f"{aa_model} (Test): {values[0]}/{values[1]}/{values[2]}/{values[3]}")
        else:
            print("No AA prediction columns found in test set.")

    # Analyze each directory group for generated text results
    for dir_path, files in file_groups.items():
        # Extract setting and dataset from path
        path_components = dir_path.split('/')
        if len(path_components) < 2:
            continue

        setting = path_components[-2]  # Setting1, Setting2, etc.
        dataset = path_components[-1]  # blog, email, etc.
        print("-"*50)
        print(f"Analyzing {setting} - {dataset}")
        print("-"*50)

        # Find test file for this dataset
        if dataset not in test_files:
            print(f"Warning: No test file found for {dataset}")
            continue
        
        # Load test data
        if setting in ['Setting1', 'Setting4']:
            test_df = pd.read_csv(test_files[dataset])
        else:
            test_df = pd.read_csv(test_files[dataset].replace("dataset_prepare", "dataset_followup"))
        
        # Process each output file for generated text results
        for output_file in files['outputs']:
            print(f"\nProcessing {output_file}")
            
            # Extract LLM model name from output file
            llm_model = os.path.basename(output_file).replace('.csv', '')
            print(f"LLM model: {llm_model}")
            
            try:
                # Load prediction data
                pred_df = pd.read_csv(output_file)
                print(f"Number of invalid samples: {pred_df['generated_length'].isna().sum()}")

                # Find AA models from prediction columns
                aa_models = [col.split('-AA-top_k-predictions')[0] 
                            for col in pred_df.columns 
                            if '-AA-top_k-predictions' in col]

                if not aa_models:
                    print(f"No AA prediction columns found.")
                    continue

                print(f"Found AA models: {', '.join(aa_models)}")
                
                # Calculate using only valid samples
                valid_mask = ~pred_df['generated_length'].isna()
                if valid_mask.any():  # Only calculate if there are valid samples
                    for aa_model in aa_models:
                        # Calculate results for generated text

                        if setting == "Setting5":
                            multiple = len(test_df)
                            summary_only = pred_df[pred_df.index < multiple].reset_index(drop=True)
                            exemplars_plus_summary = pred_df[pred_df.index >= multiple].reset_index(drop=True)

                            summary_only_valid_mask = ~summary_only['generated_length'].isna()
                            exemplars_plus_summary_valid_mask = ~exemplars_plus_summary['generated_length'].isna()

                            gen_results1 = compute_accuracies(test_df[summary_only_valid_mask], 
                                                              summary_only[summary_only_valid_mask], aa_model)
                            gen_results2 = compute_accuracies(test_df[exemplars_plus_summary_valid_mask],
                                                              exemplars_plus_summary[exemplars_plus_summary_valid_mask], aa_model)
                            
                            all_results[setting + ": SummaryOnly"][dataset][llm_model][aa_model] = gen_results1
                            all_results[setting + ": Exemplars+Summary"][dataset][llm_model][aa_model] = gen_results2
                            values1 = [gen_results1[f'top_{k}'] for k in [1, 3, 5, 10]]
                            values2 = [gen_results2[f'top_{k}'] for k in [1, 3, 5, 10]]
                            print(f"SummaryOnly: {values1[0]}/{values1[1]}/{values1[2]}/{values1[3]}")
                            print(f"Exemplars+Summary: {values2[0]}/{values2[1]}/{values2[2]}/{values2[3]}")
                        
                        elif setting == "Setting6":
                            multiple = multiple = len(test_df)
                            for i, l in enumerate([10, 8, 6, 4, 2]):
                                targets = pred_df[(pred_df.index >= i*multiple) & (pred_df.index < (i+1)*multiple)].reset_index(drop=True)
                                targets_valid_mask = ~targets['generated_length'].isna()

                                gen_results = compute_accuracies(test_df[targets_valid_mask], 
                                                                 targets[targets_valid_mask], aa_model)
                                all_results[setting + f": len={l}"][dataset][llm_model][aa_model] = gen_results
                                values = [gen_results[f'top_{k}'] for k in [1, 3, 5, 10]]
                                print(f"len={l}: {values[0]}/{values[1]}/{values[2]}/{values[3]}")

                        else:
                            gen_results = compute_accuracies(test_df[valid_mask], pred_df[valid_mask], aa_model)
                            all_results[setting][dataset][llm_model][aa_model] = gen_results
                            values = [gen_results[f'top_{k}'] for k in [1, 3, 5, 10]]
                            print(f"{aa_model} (Generated): {values[0]}/{values[1]}/{values[2]}/{values[3]}")

            except Exception as e:
                print(f"Error processing file: {str(e)}")

    # Convert nested dictionary to DataFrames
    # Original test set results
    original_rows = []
    for dataset, models in all_results['original'].items():
        for model, metrics in models.items():
            original_rows.append({
                'Dataset': dataset,
                'AA_Model': model,
                'Top_1': metrics['top_1'],
                'Top_3': metrics['top_3'],
                'Top_5': metrics['top_5'],
                'Top_10': metrics['top_10']
            })
    original_df = pd.DataFrame(original_rows)

    # Generated text results
    generated_rows = []
    for setting, datasets in all_results.items():
        if setting == 'original':  # Skip original test set results
            continue
        for dataset, llm_models in datasets.items():
            for llm_model, aa_models in llm_models.items():
                for aa_model, metrics in aa_models.items():
                    generated_rows.append({
                        'Setting': setting,
                        'Dataset': dataset,
                        'LLM_Model': llm_model,
                        'AA_Model': aa_model,
                        'Top_1': metrics['top_1'],
                        'Top_3': metrics['top_3'],
                        'Top_5': metrics['top_5'],
                        'Top_10': metrics['top_10']
                    })
    generated_df = pd.DataFrame(generated_rows)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results to CSV files
    original_df.to_csv('results/AA_original_results.csv', index=False)
    generated_df.to_csv('results/AA_generated_results.csv', index=False)
    print("\nResults saved to:")
    print("- results/AA_original_results.csv")
    print("- results/AA_generated_results.csv")

if __name__ == "__main__":
    main()