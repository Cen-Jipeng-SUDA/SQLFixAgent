import argparse
import os
import json
import time

from func_timeout import FunctionTimedOut
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.api_config import MODEL_NAME
from core.chat_manager import ChatManager
from core.const import SYSTEM_NAME
from core.utils.agent_utils import post_process, read_jsonl, jsonl_to_json, delete_file
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

    sqltool_datas = []
    _, LLM_name = os.path.split(opt.llm_path)
    if "bird" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_bird_dev.json"
        with open(pred_file, "r", encoding='utf-8') as f:
            sql_data = json.load(f)
        sqltool_datas = [sql_data[str(i)].split("\t----- bird -----\t")[0] for i in range(len(sql_data))]

    elif "spider_dev" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_dev.txt"
        with open(pred_file, "r", encoding='utf-8') as f:
            for line in f:
                sql = line.strip()
                sqltool_datas.append(sql)

    elif "spider_test" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_test.txt"
        with open(pred_file, "r", encoding='utf-8') as f:
            for line in f:
                sql = line.strip()
                sqltool_datas.append(sql)

    elif "spider_dk" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_dk.txt"
        with open(pred_file, "r", encoding='utf-8') as f:
            for line in f:
                sql = line.strip()
                sqltool_datas.append(sql)

    elif "spider_realistic" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_realistic.txt"
        with open(pred_file, "r", encoding='utf-8') as f:
            for line in f:
                sql = line.strip()
                sqltool_datas.append(sql)

    elif "spider_syn" in opt.dev_data_path:
        pred_file = LLM_name + "_pred_spider_syn.txt"
        with open(pred_file, "r", encoding='utf-8') as f:
            for line in f:
                sql = line.strip()
                sqltool_datas.append(sql)

    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    raw_dataset = json.load(open(opt.dev_data_path))

    start_time = time.time()
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
    predicted_infos=[]
    for idx, (raw_data, batch_data) in tqdm(enumerate(zip(raw_dataset, dataloader))):
        predicted_info = {"idx": idx, "generated_sql": sqltool_datas[idx], "refined_sql":None}
        filtered_items = {"schema_sequence": batch_data["schema_sequence"][0],
                          "content_sequence": batch_data["content_sequence"][0], "text": batch_data["text"][0]}
        db_path = raw_data["db_path"]
        if sqltool_datas[idx] == "SQL placeholder":
            exec_result=None
        else:
            try:
                exec_result=test_manager.sql_refiner.execute_sql(db_path,sqltool_datas[idx])
            except Exception as e:
                predicted_infos.append(predicted_info)
                continue
        msg = {
            'idx': idx,
            'db_path': db_path,
            'filtered_items': filtered_items,
            'schema_items': raw_data["schema"]["schema_items"],
            "question": raw_data["question"],
            'evidence': raw_data["evidence"],
            'exec_result': exec_result,
            'ground_truth': raw_data["sql"],
            'send_to': SYSTEM_NAME
        }
        test_manager.start(msg)
        review_pass=msg.get("review_pass",True)
        refined_sql = msg.get("refined_sql", "SQL placeholder")
        if review_pass!= True and refined_sql != "SQL placeholder":
            sqltool_datas[idx] = msg['refined_sql']
            # analysis=msg['analysis']
            predicted_info['refined_sql'] = msg['refined_sql']
        predicted_infos.append(predicted_info)
        print(sqltool_datas[idx])
        print("\n\n")

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

        for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, sqltool_datas)):
            bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
        # MODEL_NAME:api name, LLM_name:sqltool name
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_bird_dev.json"
        pred_info_file = MODEL_NAME + "_" + LLM_name + "_pred_bird_dev_info.json"
        with open(pred_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(bird_results_dict, indent=2, ensure_ascii=False))
        with open(pred_info_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(predicted_infos, indent=2, ensure_ascii=False))
        os.system("sh bird_evaluation/run_evaluation.sh " + pred_file)

    elif "spider_dev" in opt.dev_data_path:
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_dev.txt"
        pred_info_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_dev_info.json"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in sqltool_datas:
                f.write(sql + "\n")
        with open(pred_info_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(predicted_infos, indent=2, ensure_ascii=False))
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider/dev_gold.sql ' +
            '--pred ' + pred_file +
            ' --db ./data/sft_data_collections/spider/database ' +
            '--table ./data/sft_data_collections/spider/tables.json' +
            '--etype all')

    elif "spider_test" in opt.dev_data_path:
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_test.txt"
        pred_info_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_test_info.json"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in sqltool_datas:
                f.write(sql + "\n")
        with open(pred_info_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(predicted_infos, indent=2, ensure_ascii=False))
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider-test/test_data/dev_gold.sql ' +
            '--pred ' + pred_file +
            ' --db ./data/sft_data_collections/spider-test/test_database ' +
            '--table ./data/sft_data_collections/spider-test/test_data/tables.json' +
            '--etype all')

    elif "spider_dk" in opt.dev_data_path:
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_dk.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in sqltool_datas:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/Spider-DK/dk_gold.sql ' +
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database '+
            '--etype exec')

    elif "spider_realistic" in opt.dev_data_path:
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_realistic.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in sqltool_datas:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql ' +
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database ' +
            '--table ./data/sft_data_collections/spider/tables.json'+
            '--etype all')

    elif "spider_syn" in opt.dev_data_path:
        pred_file = MODEL_NAME + "_" + LLM_name + "_pred_spider_syn.txt"
        with open(pred_file, "w", encoding='utf-8') as f:
            for sql in sqltool_datas:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system(
            'python -u test_suite_sql_eval/evaluation.py ' +
            '--gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql '+
            '--pred ' + pred_file +
            '--db ./data/sft_data_collections/spider/database '+
            '--table ./data/sft_data_collections/spider/tables.json'+
            '--etype all')
