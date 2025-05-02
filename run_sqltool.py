import argparse
import os
import json
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from core.api_config import MODEL_NAME
from core.chat_manager import ChatManager
from core.utils.agent_utils import post_process
from core.utils.db_utils import check_sql_executability
from load_sft_dataset import GenerationDataset

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type=str,default=None)
    parser.add_argument('--sic_path', type=str,default=None)
    parser.add_argument('--table_num', type=int, default=6)
    parser.add_argument('--column_num', type=int, default=10)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--dev_data_path', type=str)

    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--log_path', type=str, default="./output/log.txt")
    parser.add_argument('--record_llm_path', type=str, default="./model/codes-3b")
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_option()
    print(opt)

    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    raw_dataset = json.load(open(opt.dev_data_path))

    start_time = time.time()
    predicted_sqls = []
    if "bird" in opt.dev_data_path:
        dataset_name = "bird"
    else:
        dataset_name = "spider"
    test_manager = ChatManager(train_data_path=opt.train_data_path,
                               dev_data_path=opt.dev_data_path,
                               dataset_name=dataset_name, sql_tool_path=opt.llm_path,
                               log_path=opt.log_path, record_llm_path=opt.record_llm_path)

    eval_set = GenerationDataset(
        opt.dev_data_path,
        test_manager.sqltool.tokenizer,
        max_tokens - max_new_tokens,
        "agent",
        opt.table_num,
        opt.column_num,
        opt.sic_path
    )
    # only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size=1)
    for idx, (raw_data, batch_data) in tqdm(enumerate(zip(raw_dataset, dataloader))):
        generated_sqls = test_manager.sqltool.text2sql_func_beam(batch_data["inputs"])
        generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in
                          generated_sqls]

        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, raw_data["db_path"])
            if execution_error is None:  # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            if generated_sqls[0].strip() != "":
                final_generated_sql = generated_sqls[0]
            else:
                final_generated_sql = "SQL placeholder"
        print(final_generated_sql)
        predicted_sqls.append(final_generated_sql)

    end_time = time.time()
    print("API name: {} | LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        MODEL_NAME,
        opt.llm_path,
        end_time - start_time,
        len(raw_dataset),
        (end_time - start_time) / len(raw_dataset)
    )
    )
    _, LLM_name = os.path.split(opt.llm_path)
    if "bird" in opt.dev_data_path:
        bird_results_dict = dict()
        for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
            bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
        # MODEL_NAME:api name, LLM_name:sqltool name
        pred_file = LLM_name + "_pred_bird_dev.json"
        with open(pred_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(bird_results_dict, indent=2, ensure_ascii=False))
        os.system("sh bird_evaluation/run_evaluation.sh " + pred_file)

    elif "spider_dev" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_dev.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider/dev_gold.sql ' +
            '--pred ' + pred_file +
            ' --db ./data/sft_data_collections/spider/database ' +
            '--table ./data/sft_data_collections/spider/tables.json' +
            '--etype all')

    elif "spider_test" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_test.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider-test/test_data/dev_gold.sql ' +
            '--pred ' + pred_file +
            ' --db ./data/sft_data_collections/spider-test/test_database ' +
            '--table ./data/sft_data_collections/spider-test/test_data/tables.json' +
            '--etype all')

    elif "spider_dk" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_dk.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/Spider-DK/dk_gold.sql ' +
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database ' +
            '--etype exec')


    elif "spider_realistic" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_realistic.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql ' +
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database ' +
            '--table ./data/sft_data_collections/spider/tables.json' +
            '--etype all')


    elif "spider_syn" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_syn.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql ' +
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database ' +
            '--table ./data/sft_data_collections/spider/tables.json' +
            '--etype all')

