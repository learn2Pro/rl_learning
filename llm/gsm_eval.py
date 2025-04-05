import lighteval
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
import numpy as np
from lighteval.metrics.dynamic_metrics import (
    multilingual_quasi_exact_match_metric
)
from lighteval.utils.language import Language

# gsm8k
SYSTEM_PROMPT = """
Responde in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def prompt_fn(line, task_name: str = None):
    """
    Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/default_prompts.py, or get more info
    about what this function should do in the README.
    """
    def extract_hash_answer(text:str):
        if '####' not in text:
            return None
        return text.split('####')[1].strip()
        
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[extract_hash_answer(l) for l in line["answer"]],
        gold_index=0,
    )
    
gsm_acc_metric = multilingual_quasi_exact_match_metric(
    language=Language.ENGLISH,
)


# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task = LightevalTaskConfig(
    name="gsm_acc_metric",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=['test'],
    evaluation_splits=['test'],
    few_shots_split=None,
    few_shots_select=None,
    metric=[gsm_acc_metric],  # select your metric in Metrics
)


# tasks with subset:
# community|gsm_acc_metric|0|0
TASKS_TABLE = [task]



