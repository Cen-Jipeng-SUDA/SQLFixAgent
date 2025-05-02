import json
import os
import os.path
import re

import nltk

from core.utils.db_utils import detect_special_char

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")

def parse_analysis_from_string(input_string):
    pattern = r'\{.*?\}'
    match = re.findall(pattern, input_string, re.DOTALL)[-1]
    json_dict = json.loads(match)
    analysis=json_dict['reasoning']
    label = True if 'yes' in json_dict['judgment'].lower() else False
    return label, analysis


def parse_sql_from_string(input_string):
    pattern = r'\{.*?\}'
    match = re.findall(pattern, input_string, re.DOTALL)[-1]
    json_dict = json.loads(match)
    sql = json_dict['sql']
    reasoning=json_dict['reasoning']
    return sql,reasoning


def parse_questions_from_string(input_string):
    pattern = r'\{.*?\}'
    match = re.findall(pattern, input_string, re.DOTALL)[-1]
    json_dict = json.loads(match)
    variants = json_dict['variants']
    return variants


def extract_world_info(message_dict):
    info_dict = {}
    info_dict['idx'] = message_dict.get('idx', '')
    db_id = message_dict.get('db_path', '')
    if db_id != '':
        db_id = os.path.basename(db_id).split(".")[0]
    info_dict['db_id'] = db_id
    info_dict['send_to'] = message_dict.get('send_to', '')
    info_dict['text']= message_dict.get('filtered_items', '').get('text', '')
    info_dict['evidence'] = message_dict.get('evidence', '')
    info_dict['ground_truth'] = message_dict.get('ground_truth', '')
    return info_dict


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                continue
            if json_obj is not None:
                data.append(json_obj)
    return data


def jsonl_to_json(input_file, output_file):
    data = read_jsonl(input_file)
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`" + column_name + "`")
    while "``" in sql:
        sql = sql.replace("``", "`")
    return sql


def extract_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))
    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'CD', 'SYM', 'FW', 'IN']:
            output_tokens.append("_")
        elif token in ['$', "''", '(', ')', ',', '--', '.', ':']:
            pass
        else:
            output_tokens.append(token)

    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while ("_ _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ _", "_")
    while ("_ , _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ , _", "_")

    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]

    return text_skeleton
