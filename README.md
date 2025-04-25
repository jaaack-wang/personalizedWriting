### Repo Structure 

- dataset_prepare (folder, need to download it yourself and unzip)
- notebooks (folder)
- scripts (folder)
- AA_models (folder, will be created once `train_and_eval_an_AA_model.py` is run)
- AV_models (folder, will be created once `train_and_eval_an_AV_model.py` is run)
- LLM_writing (folder, will be created once `generate_llm_writing.py` is run)
- other `.py` and `.sh` files as you can see

### Run the code

1. Create a conda environment: `conda create -n PW python 3.12.9`
2. install pip
3. run `pip install requirements.txt`
4. run some code and if there is a dependency error, fix it by installing the required package. This probably will  occur when first running `train_and_eval_an_AA_model.py` or `train_and_eval_an_AV_model.py`.