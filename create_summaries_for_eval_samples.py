import argparse
import pandas as pd
from tqdm import tqdm
from scripts.utils import get_completion
from scripts.prompt_templates import get_prompt_template_for_summarizing_text


def get_args():
    parser = argparse.ArgumentParser(description="Create summaries for evaluation samples")
    parser.add_argument("--df_fp_path", type=str, required=True, help="Path to the DataFrame file")
    parser.add_argument("--text_col", type=str, default="text", help="Column name containing the text to summarize")
    parser.add_argument("--model", type=str, default="gpt-4.1-2025-04-14", help="Model name for summarization")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for model sampling")
    parser.add_argument("--max_tries", type=int, default=5, help="Number of attempts to get a valid response")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving the DataFrame")

    return parser.parse_args()


def summarize_df(df_fp_path, text_col, 
                 prompt_tmp,
                 model="gpt-4.1-2025-04-14", 
                 temperature=0, max_tries=5, 
                 save_freq=10):
    df = pd.read_csv(df_fp_path)

    if "summary" not in df.columns:
        indices = df.index
    else:
        indices = df[df["summary"].isna()].index
        print(f"{len(df) - len(indices)} summaries already generated.")

    if len(indices) == 0:
        print("All summaries are already generated.")
        return
    
    for ix in tqdm(indices):
        text = df.at[ix, text_col]
        prompt = prompt_tmp.substitute(text=text)
        
        # Get the completion
        summary = get_completion(prompt, model=model, 
                                 temperature=temperature, 
                                 max_tries=max_tries)
        
        # Save the summary back to the DataFrame
        df.at[ix, "summary"] = summary
        
        if (ix + 1) % save_freq == 0:
            df.to_csv(df_fp_path, index=False)
    
    df.to_csv(df_fp_path, index=False)
    print("All summaries generated and saved in place.")



def main():
    args = get_args()
    print(args)
    prompt_tmp = get_prompt_template_for_summarizing_text()
    summarize_df(args.df_fp_path, 
                 args.text_col, 
                 prompt_tmp,
                 model=args.model, 
                 temperature=args.temperature, 
                 max_tries=args.max_tries, 
                 save_freq=args.save_freq)
    

if __name__ == "__main__":
    main()