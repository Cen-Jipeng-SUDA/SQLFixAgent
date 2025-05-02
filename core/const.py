MAX_ROUND = 2  # max try times of one agent talk

SYSTEM_NAME = 'System'
SQLRefiner_NAME = 'SQLRefiner'
QueryCrafter_NAME = 'QueryCrafter'
SQLReviewer_NAME = 'SQLReviewer'

similar_example_template_1 = """【Question】
{text}

【Gold SQL】
{gold_sql}
"""


similar_example_template_2 = """【Question】
{text}

【Wrong SQL】
{wrong_sql}

【Gold SQL】
{gold_sql}
"""


failure_history_template="""【Selected SQL】
{sql}
【SQLite error】
{sqlite_error}
【Exception class】
{exception_class}
"""


refine_template = """You are an experienced and professional database administrator. Given a 【Database schema】 description, a 【Database content】 and the 【Question】, you must select the most accurate query in the 【Candidate SQLs】 list. 
If there are no candidate SQL provided, generate the SQL query based on 【Database schema】 and 【Question】.
You should learn from 【Failed attempt】 before your answer.

【Similar SQL repair】
{similar_example}

Now begin your selection/generation task:
【Database schema】
{schema_sequence}
【Database content】
{content_sequence}
【Question】
{text}

【Candidate SQLs】
{candidate_sqls}

【Failed attempt】
{failure_history}

Your answer should be in correct json format:
{{
    "reasoning": ".." # The reasoning steps for selecting or generating the correct SQL query.
    "sql": ".." # The selected SQL query or generated SQL query.
}}
### Your answer:
"""

craft_case_template = """Given a natural language 【Sentence】. Please provide four different English expressions that are completely equivalent to the original one, with similar length.
【Sentence】
{query}

Please provide four variants of sentences. Your answer should be in correct json format:
{{
    "variants": [...] # List of variants 
}}

### Your answer:
"""

review_template = """You are an experienced and professional database administrator. Given a 【Database schema】 description, a 【Database content】description，and a user 【Question】 with corresponding 【SQL】, you need to start from the components of 【SQL】, step by step, to analyze whether it aligns with the user 【Question】 intent. 
【Database schema】
{schema_sequence}
【Database content】
{content_sequence}
【Question】
{text}
【SQL】
{current_sql}

Does the 【SQL】 statement match the 【Question】? Your answer should be in correct json format:
{{
    "reasoning": "..# The reasoning steps behind the answer.
    "judgment": ".." # Yes or No.
}}
### Your answer:
"""