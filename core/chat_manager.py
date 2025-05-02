import time

from core import llm
from core.agents import SQLRefiner, QueryCrafter, SQLReviewer, SQLTool
from core.const import MAX_ROUND, SYSTEM_NAME, SQLReviewer_NAME

LLM_API_FUC = llm.safe_call_llm
INIT_LOG__PATH_FUNC = llm.init_log_path


class ChatManager(object):
    def __init__(self, dataset_name: str, log_path: str, dev_data_path: str, train_data_path=None,
                 sql_tool_path=None, record_llm_path="codes-3b"):
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.dataset_name = dataset_name
        self.log_path = log_path

        self.query_crafter = QueryCrafter(dataset_name=self.dataset_name)
        self.sql_reviewer = SQLReviewer(dataset_name=self.dataset_name)
        self.sql_refiner = SQLRefiner(dev_data_path=self.dev_data_path, dataset_name=self.dataset_name,
                                      train_data_path=self.train_data_path,
                                      record_llm_path=record_llm_path)
        self.chat_group = [
            self.sql_reviewer,
            self.query_crafter,
            self.sql_refiner,
        ]
        self.sql_refiner.init_memory()
        self.sqltool = SQLTool(sql_tool_path, 256)
        self.sql_refiner.set_sql_tool(self.sqltool)
        self.query_crafter.set_sql_tool(self.sqltool)
        INIT_LOG__PATH_FUNC(log_path)

    def ping_network(self):
        print("Checking network status...", flush=True)
        try:
            _ = LLM_API_FUC("Hello world!")
            print("Network is available", flush=True)
        except Exception as e:
            raise Exception(f"Network is not available: {e}")

    def _chat_single_round(self, message: dict):
        for agent in self.chat_group:
            if message['send_to'] == agent.name:
                agent.talk(message)

    def start(self, user_message: dict):
        start_time = time.time()
        if user_message['send_to'] == SYSTEM_NAME:
            user_message['send_to'] = SQLReviewer_NAME
        for _ in range(MAX_ROUND):
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:
                break
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"\033[0;34mExecute {exec_time} seconds\033[0m", flush=True)
