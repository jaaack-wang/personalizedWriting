import pandas as pd
import os
from os import walk
from os.path import join
from collections import defaultdict

def get_test_files():
    test_files = {}
    prepare_dir = "dataset_prepare"
    
    for folder in os.listdir(prepare_dir):
        if folder.endswith("_AV_datasets"):
            dataset_type = folder.replace("_AV_datasets", "")
            test_file_path = join(prepare_dir, folder, "test.csv")
            
            if os.path.exists(test_file_path):
                test_files[dataset_type] = test_file_path
            else:
                print(f"Warning: test.csv not found in {test_file_path}")
    
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

def compute_accuracy(test_df, pred_df, model_name):
    correct = 0
    total = len(test_df)
    
    for i in range(total):
        ground_truth = test_df.iloc[i]['label']
        prediction = pred_df.iloc[i][f'{model_name}-AV-prediction']
        
        if ground_truth == prediction:
            correct += 1
    
    accuracy = (correct / total) * 100
    return round(accuracy, 2)

def main():
    test_files = get_test_files()
    file_groups = get_file_groups_by_dir("LLM_writing")

    # Dictionary to store results
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # First calculate results for original test sets
    print("\nCalculating results for original test sets...")
    for dataset, test_file in test_files.items():
        print(f"\nProcessing original test set for {dataset}")
        test_df = pd.read_csv(test_file)
        
        av_models_test = [col.split('-AV-prediction')[0].split('/')[-1]
                            for col in test_df.columns 
                            if '-AV-prediction' in col]

        if av_models_test:
            print(f"Found AV models in test set: {', '.join(av_models_test)}")
            for av_model in av_models_test:
                test_accuracy = compute_accuracy(test_df, test_df, av_model)
                all_results['original'][dataset][av_model] = test_accuracy
                print(f"{av_model}: {test_accuracy}%")
        else:
            print("No AV prediction columns found in test set.")

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
        test_df = pd.read_csv(test_files[dataset])
        # if setting in ['Setting1', 'Setting4']:
        #     test_df = pd.read_csv(test_files[dataset])
        # else:
        #     test_df = pd.read_csv(test_files[dataset].replace("dataset_prepare", "dataset_followup"))
        
        # Process each output file
        for output_file in files['outputs']:
            print(f"\nProcessing {output_file}")
            
            # Extract LLM model name from output file
            llm_model = os.path.basename(output_file).replace('.csv', '')
            print(f"LLM model: {llm_model}")
            
            try:
                # Load prediction data
                pred_df = pd.read_csv(output_file)
                print(f"Number of invalid samples: {pred_df['generated_length'].isna().sum()}")

                # Find AV models from prediction columns
                av_models = [col.split('-AV-prediction')[0] 
                            for col in pred_df.columns 
                            if '-AV-prediction' in col]

                if not av_models:
                    print(f"No AV prediction columns found.")
                    continue

                print(f"Found AV models: {', '.join(av_models)}")
                
                # Calculate using only valid samples
                valid_mask = ~pred_df['generated_length'].isna()
                # =========== Changes by Jack ===========
                # pred_df = pred_df[valid_mask] 
                for av_model in av_models:
                    if f"{av_model}-AV-prediction" in pred_df.columns:

                        if setting == "Setting5":
                            multiple = len(pred_df) // 2
                            
                            summary_only = pred_df[pred_df.index < multiple].reset_index(drop=True)
                            summary_only = summary_only[~summary_only['generated_length'].isna()]

                            exemplars_plus_summary = pred_df[pred_df.index >= multiple].reset_index(drop=True)
                            exemplars_plus_summary = exemplars_plus_summary[~exemplars_plus_summary['generated_length'].isna()]

                            accuracy1 = summary_only[f"{av_model}-AV-prediction"].mean() * 100 
                            accuracy1 = round(accuracy1, 2)
                            accuracy2 = exemplars_plus_summary[f"{av_model}-AV-prediction"].mean() * 100
                            accuracy2 = round(accuracy2, 2)
                            all_results[setting + ": SummaryOnly"][dataset][llm_model][av_model] = accuracy1
                            all_results[setting + ": Exemplars+Summary"][dataset][llm_model][av_model] = accuracy2
                            print(f"SummaryOnly: {accuracy1}%")
                            print(f"Exemplars+Summary: {accuracy2}%")


                        elif setting == "Setting6":
                            multiple = len(pred_df) // 5
                            for i, l in enumerate([10, 8, 6, 4, 2]):
                                targets = pred_df[(pred_df.index >= i*multiple) & (pred_df.index < (i+1)*multiple)].reset_index(drop=True)
                                sub = targets[~targets['generated_length'].isna()]
                                accuracy = sub[f"{av_model}-AV-prediction"].mean() * 100
                                accuracy = round(accuracy, 2)
                                all_results[setting + f": len={l}"][dataset][llm_model][av_model] = accuracy
                                print(f"len={l}: {accuracy}%")


                        else:
                            accuracy = pred_df[valid_mask][f"{av_model}-AV-prediction"].mean() * 100 
                            accuracy = round(accuracy, 2)
                            all_results[setting][dataset][llm_model][av_model] = accuracy
                            print(f"{av_model}: {accuracy}%")

                # if valid_mask.any():  # Only calculate if there are valid samples
                #     for av_model in av_models:
                #         accuracy = compute_accuracy(test_df, pred_df[valid_mask], av_model)
                #         if accuracy is not None:
                #             all_results[setting][dataset][llm_model][av_model] = accuracy
                #             print(f"{av_model}: {accuracy}%")

            except Exception as e:
                print(f"Error processing file: {str(e)}")

    # Convert nested dictionary to DataFrames
    # Original test set results
    original_rows = []
    for dataset, models in all_results['original'].items():
        for model, accuracy in models.items():
            original_rows.append({
                'Dataset': dataset,
                'AA_Model': model,
                'Top_1': accuracy
            })
    original_df = pd.DataFrame(original_rows)

    # Generated text results
    generated_rows = []
    for setting, datasets in all_results.items():
        if setting == 'original':  # Skip original test set results
            continue
        for dataset, llm_models in datasets.items():
            for llm_model, aa_models in llm_models.items():
                for aa_model, accuracy in aa_models.items():
                    generated_rows.append({
                        'Setting': setting,
                        'Dataset': dataset,
                        'LLM_Model': llm_model,
                        'AV_Model': aa_model, # =============== Jack: Changed from AA_Model to AV_Model
                        'Top_1': accuracy
                    })
    generated_df = pd.DataFrame(generated_rows)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results to CSV files
    original_df.to_csv('results/AV_original_results.csv', index=False)
    generated_df.to_csv('results/AV_generated_results.csv', index=False)
    print("\nResults saved to:")
    print("- results/AV_original_results.csv")
    print("- results/AV_generated_results.csv")


if __name__ == "__main__":
    main() 