import json
import os
import abc
import logging
import sqlite3

import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from simcse import SimCSE
import core.llm as llm
from core.utils.agent_utils import (
    parse_sql_from_string,
    extract_world_info,
    parse_questions_from_string,
    parse_analysis_from_string,
    post_process,
    read_jsonl,
    extract_skeleton
)
from load_sft_dataset import GenerationDataset, prepare_inputs
from core.const import *

LLM_API_FUC = llm.safe_call_llm
llm.init_log_path("./output/log.txt")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class SQLTool:

    def __init__(self, llm_path: str, max_new_tokens):
        self._llm_path = llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self._model = AutoModelForCausalLM.from_pretrained(llm_path, device_map='auto',
                                                           torch_dtype=torch.float16).eval()
        self._max_new_tokens = max_new_tokens

    def text2sql_func_beam(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                num_beams=4,
                num_return_sequences=4
            )
        generated_sqls = self.tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

        return generated_sqls

    def text2sql_func(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )
        generated_sqls = self.tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

        return generated_sqls


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self, dataset_name):
        self.sqltool = None
        self.dataset_name = dataset_name

    @abc.abstractmethod
    def talk(self, message: dict):
        pass

    @func_set_timeout(120)
    def execute_sql(self, db_path, sql: str) -> dict:
        # Get database connection
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            return {
                "sql": str(sql),
                "data": result[:5],
                "sqlite_error": "",
                "exception_class": ""
            }
        except sqlite3.Error as er:
            return {
                "sql": str(sql),
                "sqlite_error": str(' '.join(er.args)),
                "exception_class": str(er.__class__)
            }
        except Exception as e:
            return {
                "sql": str(sql),
                "sqlite_error": str(e.args),
                "exception_class": str(type(e).__name__)
            }

    def is_need_fix(self, exec_result: dict):
        if exec_result is None:
            return True
        # spider exist dirty values, even gold sql execution result is None
        if self.dataset_name == 'spider':
            if 'data' not in exec_result:
                return True
            return False

        data = exec_result.get('data', None)
        if data is not None:
            if len(data) == 0:
                exec_result['sqlite_error'] = 'no data selected'
                return True
            for t in data:
                for n in t:
                    if n is None:
                        exec_result['sqlite_error'] = 'exist None value, you can add `NOT NULL` in SQL'
                        return True
            return False
        return True

    def set_sql_tool(self, sqltool):
        self.sqltool = sqltool

class SQLReviewer(BaseAgent):
    """
    Review the generated SQL.
    """

    def __init__(self, dataset_name):
        super().__init__(dataset_name=dataset_name)
        self._message = {}
        self.name = SQLReviewer_NAME

    def talk(self, message: dict):
        """
        :param message: {
                         "filtered_items": filtered_items,
                         "exec_result":exec_result
                        }
        :return: review_pass, analysis
        """
        if message['send_to'] != self.name:
            return
        self._message = message
        logger.info('SQLReviewer is working...')
        filtered_items = message['filtered_items']
        exec_result = message['exec_result']
        if exec_result is None:
            message['review_analysis'] = "Execution error"
            message['review_pass'] = False
            message['send_to']=SQLRefiner_NAME
            logger.info('Found SQL placeholder...')
            return
        if self.is_need_fix(exec_result):
            message['review_analysis'] = "Execution error"
            message['review_pass'] = False
            message['send_to'] = QueryCrafter_NAME
            logger.info('Found execution error...')
            return
        current_sql = message["exec_result"]["sql"]


        prompt = review_template.format(schema_sequence=filtered_items["schema_sequence"],
                                        content_sequence=filtered_items["content_sequence"],
                                        text=filtered_items["text"],
                                        current_sql=current_sql)

        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info).strip()

        try:
            label, analysis = parse_analysis_from_string(reply)
        except Exception as e:
            logging.error('Parse analysis error: %s', str(e))
            message['review_pass'] = True
            message['send_to'] = SYSTEM_NAME
            return

        message['review_pass'] = label
        message['review_analysis'] = analysis

        if label:
            message['send_to'] = SYSTEM_NAME
            logging.info("No error found!")
        else:
            message['send_to'] = QueryCrafter_NAME
            logging.info("Error detect!")


class SQLRefiner(BaseAgent):
    """
    Assist in Repairing SQL statements.
    """

    def __init__(self, train_data_path: str, dev_data_path: str, dataset_name: str, record_llm_path):
        super().__init__(dataset_name=dataset_name)
        self.name = SQLRefiner_NAME
        self.dev_data_path = dev_data_path
        self.train_data_path = train_data_path
        self._message = {}
        self.record_llm_path = record_llm_path

        self.exec_error_set = None
        self.exec_error_similarities = None

    def _record_exec_error_to_file(self, record):
        if not os.path.exists("./core/memory"):
            os.makedirs("./core/memory")

        record_file = os.path.join("./core/memory", self.dataset_name + "_exec_error_record.jsonl")
        with open(record_file, "a") as f:
            json.dump(record, f)
            f.write('\n')

    def init_memory(self):
        if self.dataset_name == 'bird':
            memory_path = "./core/memory/bird_exec_error_record.jsonl"
        else:
            memory_path = "./core/memory/spider_exec_error_record.jsonl"

        if not os.path.exists(memory_path):
            if self.train_data_path is not None:
                self._record_exec_error(self.train_data_path)
            else:
                return

        records = read_jsonl(memory_path)
        error_records = []
        for record in records:
            error_records.append(record)
        self.exec_error_set = error_records
        eval_set = json.load(open(self.dev_data_path))
        eval_set_questions = [data["question"] for data in eval_set]
        eval_set_question_skeletons = [extract_skeleton(question) for question in eval_set_questions]

        exec_error_set_questions = [data["question"] for data in self.exec_error_set]
        exec_error_set_question_skeletons = [extract_skeleton(question) for question in
                                             exec_error_set_questions]
        simsce_model = SimCSE("./model/sup-simcse-roberta-base")
        question_similarities = simsce_model.similarity(eval_set_questions, exec_error_set_questions)
        question_skeleton_similarities = simsce_model.similarity(eval_set_question_skeletons,
                                                                 exec_error_set_question_skeletons)
        self.exec_error_similarities = np.maximum(question_similarities, question_skeleton_similarities)

        del simsce_model

    def _retrieve_exec_error(self, retrieve_idx, k):
        similarity = self.exec_error_similarities[retrieve_idx]
        top_k_indices = sorted(range(len(similarity)), key=lambda x: similarity[x], reverse=True)[:k]
        retrieve_set = []
        for i in top_k_indices:
            retrieve_set.append(self.exec_error_set[i])
        return retrieve_set

    def _record_exec_error(self, text2sql_data_dir):
        raw_dataset = json.load(open(text2sql_data_dir))
        if self.dataset_name == "bird":
            sic_path = "./sic_ckpts/sic_bird_with_evidence"
        else:
            sic_path = "./sic_ckpts/sic_spider"

        sqltool = SQLTool(self.record_llm_path, 256)
        eval_set = GenerationDataset(
            text2sql_data_dir,
            sqltool.tokenizer,
            4096 - 256,
            "agent",
            6,
            10,
            sic_path
        )
        dataloader = DataLoader(eval_set, batch_size=1)
        for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
            generated_sql = sqltool.text2sql_func(batch_data["inputs"])[0]
            generated_sql = post_process(generated_sql, raw_data["schema"]["schema_items"])
            logging.debug('Generated SQL statement: %s', generated_sql)
            db_id = os.path.basename(raw_data["db_path"]).split(".")[0]
            is_timeout = False
            exec_result = None
            try:
                exec_result = self.execute_sql(raw_data["db_path"],generated_sql)
            except Exception as e:
                is_timeout = True
            except FunctionTimedOut as fto:
                is_timeout = True

            if is_timeout:
                continue

            if self.is_need_fix(exec_result):
                record = {"question": raw_data["question"], "db_id": db_id, "sql": raw_data["sql"],
                          "filtered_items": {"schema_sequence": batch_data["schema_sequence"][0],
                                             "content_sequence": batch_data["content_sequence"][0],
                                             "text": batch_data["text"][0]},
                          "exec_result": exec_result}
                self._record_exec_error_to_file(record)

            logging.info('Execution result: %s', exec_result)
        del sqltool

    def talk(self, message: dict):
        """
        :param message: {
                         "filtered_items": filtered_items,
                         "schema_items": schema_items,
                         "sql_variants": sql_variants,
                         "exec_result":exec_result,
                         "idx"  #Used for similarity retrieval
                         "db_path": db_path
                        }
        :return: refined_sql
        """
        if message['send_to'] != self.name:
            return
        logger.info('SQLRefiner is working...')
        self._message = message
        exec_result = message['exec_result']
        schema_sequence = message["filtered_items"]["schema_sequence"]
        content_sequence = message["filtered_items"]["content_sequence"]
        text = message["filtered_items"]["text"]
        schema_items=message['schema_items']
        db_path=message['db_path']
        idx = message.get("idx", '')
        sql_variants = message.get("sql_variants", [])
        if self.is_need_fix(exec_result) is False:
            sql_variants.append(exec_result['sql'])
        sql_variants=list(set(sql_variants))
        if len(sql_variants)==1:
            message['send_to'] = SYSTEM_NAME
            message['refined_sql'] = sql_variants[0]
            logging.info('Only one refined sql, refine done: %s', message['refined_sql'])
            return

        logging.info('Sql variants: %s', sql_variants)

        if idx == '' or self.exec_error_set is None:
            logger.info('Memory is not initialized.')
            similar_example_prompt = "None"
        else:
            similar_example = self._retrieve_exec_error(idx, 1)[0]
            # similar_example_prompt = similar_example_template_2.format(
            #     text=similar_example["filtered_items"]["text"],
            #     wrong_sql=similar_example["exec_result"]["sql"], gold_sql=similar_example["sql"])
            similar_example_prompt = similar_example_template_1.format(
                text=similar_example["filtered_items"]["text"],
                gold_sql=similar_example["sql"])

        res="SQL placeholder"
        refined_exec_result = None
        failure_history = ""
        message['refined_label']=False
        times=3
        for i in range(times):
            if refined_exec_result is not None:
                failure_history += failure_history_template.format(sql=refined_exec_result["sql"],
                                                               sqlite_error=refined_exec_result["sqlite_error"],
                                                               exception_class=refined_exec_result["exception_class"]) + "\n"
            prompt = refine_template.format(similar_example=similar_example_prompt,
                                            schema_sequence=schema_sequence,
                                            content_sequence=content_sequence,
                                            text=text,
                                            candidate_sqls=str(sql_variants),
                                            failure_history=failure_history)
            word_info = extract_world_info(self._message)
            reply = LLM_API_FUC(prompt, **word_info).strip()
            try:
                res,reasoning = parse_sql_from_string(reply)
                res = post_process(res, schema_items)
            except Exception as e:
                logging.error('Parse SQL error: %s', str(e))
                refined_exec_result=None
                continue
            is_timeout = False
            try:
                refined_exec_result = self.execute_sql(db_path,res)
            except Exception as e:
                is_timeout = True
                refined_exec_result = None
            except FunctionTimedOut as fto:
                is_timeout = True
                refined_exec_result = None

            is_need = self.is_need_fix(refined_exec_result)
            if not is_need or is_timeout:  # correct in one pass or refine success or timeout
                message['refined_label']=True
                break
        message['send_to'] = SYSTEM_NAME
        if message['refined_label']:
            message['refined_sql'] = res
            logging.info('Refined sql: %s', message['refined_sql'])
        elif len(sql_variants)!=0:
            message['refined_sql'] = sql_variants[-1]
            logging.info('Refine failed, replace with: %s', message['refined_sql'])
        else:
            message['refined_sql'] = "SQL placeholder"
            logging.info('Refine failed')



class QueryCrafter(BaseAgent):
    """
    Write similar test cases.
    """

    def __init__(self, dataset_name):
        super().__init__(dataset_name=dataset_name)
        self._message = {}
        self.name = QueryCrafter_NAME

    def talk(self, message: dict):
        """
        :param message: {
                         "filtered_items": filtered_items,
                         "schema_items":schema_items,
                         "db_path":db_path
                         "evidence":evidence
                         "question":question
                        }
        :return: refined_sql
        """
        if message['send_to'] != self.name:
            return
        logger.info('QueryCrafter is working...')
        self._message = message
        filtered_items = message['filtered_items']
        db_path = message['db_path']
        evidence = message['evidence']
        question = message['question']
        schema_items=message['schema_items']
        prompt = craft_case_template.format(query=question)

        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info).strip()
        try:
            res = parse_questions_from_string(reply)
        except Exception as e:
            logging.error('parse questions error: %s', str(e))
            message['sql_variants'] = []
            message['send_to'] = SQLRefiner_NAME
            return

        sql_variants=[]
        if isinstance(res, list) and len(res) > 0:
            logging.info("generating sql of query variants...")
            for question in res:
                text = evidence + " " + question \
                    if evidence != "" else question
                prefix_seq = filtered_items["schema_sequence"] + "\n" + filtered_items[
                    "content_sequence"] + "\n" + text + "\n"
                tool_input = prepare_inputs(prefix_seq, self.sqltool.tokenizer, 4096 - 256)
                tool_input["input_ids"] = tool_input["input_ids"].unsqueeze(0)
                tool_input["attention_mask"] = tool_input["attention_mask"].unsqueeze(0)
                generated_sql = self.sqltool.text2sql_func(tool_input)[0]
                generated_sql = post_process(generated_sql, schema_items)

                logging.info('Sql variant: %s', str(generated_sql))
                exec_result = None
                try:
                    exec_result = self.execute_sql(db_path, generated_sql)
                except Exception as e:
                    pass
                is_need_fix=self.is_need_fix(exec_result)
                if is_need_fix:
                    logging.info('Sql variant exec error')
                else:
                    sql_variants.append(generated_sql)

        message['sql_variants'] = sql_variants
        message['send_to'] = SQLRefiner_NAME

