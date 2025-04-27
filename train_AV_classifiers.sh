# ================== allenai/longformer-base-4096 ==================

python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/blog_AV_datasets --num_train_epochs=10 --model_name=allenai/longformer-base-4096 --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/CCAT50_AV_datasets --num_train_epochs=10 --model_name=allenai/longformer-base-4096 --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/enron_AV_datasets --num_train_epochs=10 --model_name=allenai/longformer-base-4096 --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/reddit_AV_datasets --num_train_epochs=10 --model_name=allenai/longformer-base-4096 --max_length=2048


# ================== answerdotai/ModernBERT-base ==================

python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/blog_AV_datasets --num_train_epochs=10 --model_name=answerdotai/ModernBERT-base --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/CCAT50_AV_datasets --num_train_epochs=10 --model_name=answerdotai/ModernBERT-base --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/enron_AV_datasets --num_train_epochs=10 --model_name=answerdotai/ModernBERT-base --max_length=2048
python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/reddit_AV_datasets --num_train_epochs=10 --model_name=answerdotai/ModernBERT-base --max_length=2048




