import os

import pandas as pd
from datasets import Dataset


def load_dataset(df):
	ls = []
	for i in df['retrieved_contexts']:
		ls.append([i])
	df['retrieved_contexts'] = ls
	return Dataset.from_pandas(df)


os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://api.bianxie.ai/v1"
model_name="qwen-max"
file_name=f"eval_data/res-{model_name}.csv"
da = pd.read_csv(file_name)
from ragas import evaluate
from ragas.metrics import (
	faithfulness,
	context_recall,
	answer_correctness,
	context_precision
)

dataset = load_dataset(da)

metrics = [
	faithfulness,
	context_recall,
	context_precision,
	answer_correctness
]
results = evaluate(
	dataset=dataset,
	metrics=metrics,
)
file_name_res=f"eval_data/{model_name}_evaluation.csv"
results.to_pandas().to_csv(file_name_res)
print(results)
print(sum(results["faithfulness"])/len(results["faithfulness"]))
print(sum(results["answer_correctness"])/len(results["answer_correctness"]))