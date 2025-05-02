set -e
## --------------- Create the Result of CodeS-3b --------------- #
## Bird's dev
CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --train_data_path ./data/sft_bird_with_evidence_train_text2sql.json --dev_data_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's test
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_test_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-DK
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Syn
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Realistic
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## --------------- Fix the Result of CodeS-3b --------------- #
## Bird's dev
CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --train_data_path ./data/sft_bird_with_evidence_train_text2sql.json --dev_data_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's test
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_test_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-DK
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Syn
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Realistic
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
#
## --------------- Create the Result of CodeS-7b --------------- #
## Bird's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --train_data_path ./data/sft_bird_with_evidence_train_text2sql.json --dev_data_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's test
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_test_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-DK
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Syn
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Realistic
#CUDA_VISIBLE_DEVICES=0 python -u run_sqltool.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## --------------- Fix the Result of CodeS-7b --------------- #
## Bird's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --train_data_path ./data/sft_bird_with_evidence_train_text2sql.json --dev_data_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's dev
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider's test
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_test_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-DK
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Syn
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256
#
## Spider-Realistic
#CUDA_VISIBLE_DEVICES=0 python -u run_fix.py --llm_path ./model/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --train_data_path ./data/sft_spider_train_text2sql.json --dev_data_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256
