python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/enron_train.csv --test_df_fp=dataset_prepare/enron_test.csv --num_train_epochs=20 --model_name=answerdotai/ModernBERT-base
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/CCAT50_train.csv --test_df_fp=dataset_prepare/CCAT50_test.csv --num_train_epochs=20 --model_name=answerdotai/ModernBERT-base
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/reddit_train.csv --test_df_fp=dataset_prepare/reddit_test.csv --num_train_epochs=20 --model_name=answerdotai/ModernBERT-base
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/blog_train.csv --test_df_fp=dataset_prepare/blog_test.csv --num_train_epochs=20 --model_name=answerdotai/ModernBERT-base


python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/enron_train.csv --test_df_fp=dataset_prepare/enron_test.csv --num_train_epochs=20 --model_name=allenai/longformer-base-4096
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/CCAT50_train.csv --test_df_fp=dataset_prepare/CCAT50_test.csv --num_train_epochs=20 --model_name=allenai/longformer-base-4096
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/reddit_train.csv --test_df_fp=dataset_prepare/reddit_test.csv --num_train_epochs=20 --model_name=allenai/longformer-base-4096
python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/blog_train.csv --test_df_fp=dataset_prepare/blog_test.csv --num_train_epochs=20 --model_name=allenai/longformer-base-4096
