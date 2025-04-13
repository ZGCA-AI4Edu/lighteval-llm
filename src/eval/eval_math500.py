"""
评估 MATH-500 数据集
"""
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import numpy as np


########################################################
# prompt_fn
########################################################
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}

""".strip()

DETAILED_QUERY_TEMPLATE = """
Answer the problem with a detailed thinking process:

{Question}

""".strip()

BRIEF_QUERY_TEMPLATE = """
Answer the problem with a brief thinking process:

{Question}

""".strip()

MIDDLE_QUERY_TEMPLATE = """
Answer the problem with a appropriate (between detailed and brief) thinking process:

{Question}

""".strip()

def math500_prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/default_prompts.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


########################################################
# metric：latex_gold_metric
########################################################
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)


########################################################
# metric：mean_length_metric
########################################################
def compute_sample_length(predictions: list[str], **kwargs) -> float:
    """计算模型 response的长度（以字符为单位）"""
    response = predictions[0]
    return len(response)


def agg_mean_length(items: list[float], **kwargs) -> float:
    flat_items = [item for item in items]
    return float(np.mean(flat_items))


mean_length_metric = SampleLevelMetric(
    metric_name="mean_length",
    higher_is_better=False,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=compute_sample_length,
    corpus_level_fn=agg_mean_length,
)


########################################################
# task
########################################################
math500_task = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    trust_dataset=True,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric, mean_length_metric],
    version=1,
)


TASKS_TABLE = [math500_task]


if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
