## ------------ MAIN EXPERIMENTS ------------

# ================== gpt-4o-mini ==================

# setting=1
# LLM="openai/gpt-4o-mini-2024-07-18"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# setting=4
# LLM="openai/gpt-4o-mini-2024-07-18"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# ================== gpt-4o ==================

# setting=1
# LLM="openai/gpt-4o-2024-08-06"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# setting=4
# LLM="openai/gpt-4o-2024-08-06"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# ================== gemini-2.0-flash ==================
# setting=1 ### Done (A few cases not generated due to safe guardrails)
# LLM="gemini/gemini-2.0-flash"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# setting=4
# LLM="gemini/gemini-2.0-flash"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# ================== anthropic/claude-3-5-haiku-20241022 ==================
# setting=1
# LLM="anthropic/claude-3-5-haiku-20241022"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM


# setting=4
# LLM="anthropic/claude-3-5-haiku-20241022"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM


# ================== gemma-3-27b-it ==================

# setting=1
# LLM="gemini/gemma-3-27b-it"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM


# setting=4
# LLM="gemini/gemma-3-27b-it"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM


# ================== deepseek/deepseek-chat ==================

# setting=1
# LLM="deepseek/deepseek-chat"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM



# ================== groq/meta-llama/llama-4-maverick-17b-128e-instruct ==================

# setting=1
# LLM="groq/meta-llama/llama-4-maverick-17b-128e-instruct"
# python generate_llm_writing.py --training_df_fp=dataset_prepare/enron_train.csv --evaluation_df_fp=dataset_prepare/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/CCAT50_train.csv --evaluation_df_fp=dataset_prepare/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/reddit_train.csv --evaluation_df_fp=dataset_prepare/reddit_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM



## ------------ FOLLOWUP EXPERIMENTS ------------


# ================== gpt-4o ==================

# setting=2
# LLM="openai/gpt-4o-2024-08-06"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# setting=3
# LLM="openai/gpt-4o-2024-08-06"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# setting=6
# LLM="openai/gpt-4o-2024-08-06"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM



# ================== gemini-2.0-flash ==================


# setting=2
# LLM="gemini/gemini-2.0-flash"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# setting=3
# LLM="gemini/gemini-2.0-flash"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# setting=6
# LLM="gemini/gemini-2.0-flash"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# ================== groq/meta-llama/llama-4-maverick-17b-128e-instruct ==================

# setting=2
# LLM="groq/meta-llama/llama-4-maverick-17b-128e-instruct"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


# setting=3
# LLM="groq/meta-llama/llama-4-maverick-17b-128e-instruct"
# python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
# python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM


setting=6
LLM="groq/meta-llama/llama-4-maverick-17b-128e-instruct"
python generate_llm_writing.py --training_df_fp=dataset_followup/blog_train.csv --evaluation_df_fp=dataset_followup/blog_test.csv --setting=$setting --llm=$LLM
python generate_llm_writing.py --training_df_fp=dataset_followup/CCAT50_train.csv --evaluation_df_fp=dataset_followup/CCAT50_test.csv --setting=$setting --llm=$LLM
python generate_llm_writing.py --training_df_fp=dataset_followup/enron_train.csv --evaluation_df_fp=dataset_followup/enron_test.csv --setting=$setting --llm=$LLM
python generate_llm_writing.py --training_df_fp=dataset_followup/reddit_train.csv --evaluation_df_fp=dataset_followup/reddit_test.csv --setting=$setting --llm=$LLM