## Dataset

- https://drive.google.com/drive/folders/1pL3Mb0y4xPNYbKG3nB1bDIPG_6mSr3C4?usp=sharing
    - Nafis: Each has a train and test portion (50-50 split). I limit the author number to 100 for each dataset. Word limit is maintained 50/100-1000/1500. Also, the formatting is the same for all: author and text are two major columns. I keep some other columns since they might be useful later for prompt generation or clustering."
    - It seems that the ranges for the word limit are not inclusive.


#### Questions

1. The word count is estimated by using nltk's default word tokenizer. However, LLMs' tokenizers are vastly different from that. Is that ok?
2. For the AV classifier, should I train one for each dataset or should I train one using all the datasets available for the training?


## To Do

1. LLM Prompting Pipeline
2. Clustering pipeline 
3. Authorship Verification pipeline
    - Automatically generate data for AV
    - Finetuning a AV classifier based on long-former
4. 
