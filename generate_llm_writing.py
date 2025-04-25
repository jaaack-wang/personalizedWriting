import os
import argparse
import pandas as pd
from tqdm import tqdm
from scripts.utils import get_completion

from scripts.utils import (
    count_words,
    round_up_to_nearest_10,
    list_writing_samples
)

from scripts.prompt_templates import (
 get_prompt_template_for_writing_setting1, 
 get_prompt_template_for_writing_setting4   
)


def get_args():
    parser = argparse.ArgumentParser(description="Create writing prompts and prompt LLMs to generate writing.")
    parser.add_argument("--evaluation_df_fp", type=str, required=True, help="Path to the evaluation DataFrame.")
    parser.add_argument("--llm", type=str, required=True, help="LLM model to use for generation. Use litellm name convention.")

    parser.add_argument("--training_df_fp", type=str, default=None, help="Path to the training DataFrame. Default is None.")
    parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4, 5], default=1, help="Prompt setting (1-5). Default is 1.")
    parser.add_argument("--genre", type=str, default=None, help="Genre of the writing samples. Default is None (auto infer from dataset if possible).")
    parser.add_argument("--author_col", type=str, default="author", help="Column name for author in the DataFrame. Default is 'author'.")
    parser.add_argument("--text_col", type=str, default="text", help="Column name for text in the DataFrame. Default is 'text'.")
    parser.add_argument("--summary_col", type=str, default="summary", help="Column name for summary in the DataFrame. Default is 'summary'.")
    parser.add_argument("--num_exemplars", type=int, default=5, help="Number of exemplars per author. Default is 5.")
    
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the LLM. Default is 0.")
    parser.add_argument("--max_tries", type=int, default=5, help="Number of tries for LLM completion. Default is 5.")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving LLM outputs. Default is 10.")

    return parser.parse_args()


def create_writing_prompts_setting1(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    num_exemplars=5):
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert training_df[author_col].value_counts().min() >= num_exemplars, \
        f"Each author must have at least {num_exemplars} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting1()        
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        
        author = row[author_col]
        summary = row[summary_col]
        
        num_words = round_up_to_nearest_10(count_words(row[text_col]))
        samples = training_df[training_df[author_col]==author][text_col].sample(num_exemplars)
        writing_samples = list_writing_samples(samples)
        prompt = prompt_tmp.substitute(writing_samples=writing_samples, 
                                       genre=genre, num_words=num_words,
                                       summary=summary)
        evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
        evaluation_df.at[ix, "prompt"] = prompt

    evaluation_df.to_csv(evaluation_df_fp, index=False)
    
    return evaluation_df


def create_writing_prompts_setting4(evaluation_df_fp, 
                                    genre,
                                    text_col="text", 
                                    summary_col="summary"):
    
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."
    
    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting4()
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        summary = row[summary_col]
        
        num_words = round_up_to_nearest_10(count_words(row[text_col]))
        prompt = prompt_tmp.substitute(genre=genre, num_words=num_words,
                                       summary=summary)
        evaluation_df.at[ix, "prompt"] = prompt

    evaluation_df.to_csv(evaluation_df_fp, index=False)

    return evaluation_df


def generate_or_load_writing_prompts(args, dire):
    if os.path.exists(os.path.join(dire, "prompts.csv")):
        print(f"Prompts already exist in {dire}/prompts.csv")
        df = pd.read_csv(os.path.join(dire, "prompts.csv"))

        return df
    
    if args.setting == 1:

        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 1."
        
        df = create_writing_prompts_setting1(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            num_exemplars=args.num_exemplars
        )

    elif args.setting == 4:
        df = create_writing_prompts_setting4(
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            text_col=args.text_col,
            summary_col=args.summary_col
        )
    else:

        raise ValueError("Setting not implemented yet.")
    
    df.to_csv(os.path.join(dire, "prompts.csv"), index=False)
    print(f"Prompts saved to {dire}/prompts.csv")

    return df


def prompt_llm_to_generate_writing(df, save_dir, model, 
                                   temperature=0, max_tries=5, 
                                   save_freq=10):
    model_name = model.split("/")[-1]
    fp = os.path.join(save_dir, model_name + ".csv")

    if os.path.exists(fp):
        llm_df = pd.read_csv(fp)
        llm_df.writing = llm_df.writing.replace("SOMETHING_WRONG", pd.NA)
        num_already_generated = llm_df.writing.notna().sum()
        if len(df) == num_already_generated:
            print(f"All writing samples already generated for {model_name}.")
            return
        else:
            print(f"{num_already_generated} writing samples generated for {model_name}. Continuing...")
            indices = sorted(set(df.index) - set(llm_df[llm_df.writing.notna()].index))
    else:
        indices = list(range(len(df)))
        llm_df = pd.DataFrame(columns=["writing"])
    
    for j, ix in tqdm(enumerate(indices), total=len(indices)):
        prompt = df.at[ix, "prompt"]
        completion = get_completion(prompt, model=model, 
                                    temperature=temperature, 
                                    max_tries=max_tries)
        llm_df.at[ix, "writing"] = completion

        if (j+1) % save_freq == 0:
            llm_df.to_csv(fp, index=False)
    
    llm_df.to_csv(fp, index=False)
    print(f"Writing generated and saved to {fp}")


def main():
    args = get_args()
    print(args)
    
    dataset = args.evaluation_df_fp.split("/")[-1].split(".")[0].split("_")[0]
    dire = f"LLM_writing/Setting{args.setting}/{dataset}"
    os.makedirs(dire, exist_ok=True)

    if args.training_df_fp is not None:
        dataset_ = args.training_df_fp.split("/")[-1].split(".")[0].split("_")[0]
        assert dataset == dataset_, \
            f"Training and evaluation datasets must be the same. {dataset} != {dataset_}"

    if args.genre is None:
        if dataset.startswith("blog"):
            args.genre = "blog post"
        elif dataset.startswith("enron"):
            args.genre = "email"
        elif dataset.startswith("reddit"):
            args.genre = "reddit post"
        elif dataset.startswith("CCAT50"):
            args.genre = "news article"
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Please specify a genre.")

    #### generating or loading prompts
    df = generate_or_load_writing_prompts(args, dire)

    #### prompting llm to generate writing
    prompt_llm_to_generate_writing(
        df, 
        save_dir=dire, 
        model=args.llm,
        temperature=args.temperature,
        max_tries=args.max_tries,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()
