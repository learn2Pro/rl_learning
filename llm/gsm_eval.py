import lighteval
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)

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
    """Defines how to go from a dataset line to a doc object.
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
        choices=[f" {c}" for c in line["choices"]],
        gold_index=extract_hash_answer(line),
        instruction=SYSTEM_PROMPT,
    )

custom_metric = SampleLevelMetric(
    metric_name="my_custom_metric_name",
    higher_is_better=True,
    category=MetricCategory.IGNORED,
    use_case=MetricUseCase.NONE,
    sample_level_fn=lambda x: x,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # How to aggreagte the samples metrics
)