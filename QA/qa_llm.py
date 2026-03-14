import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain, LLMChain, SequentialChain, TransformChain
api_keys={
	"Baichuan4-Turbo":"",
	"deepseek-v3":"",
	"qwen-max":"",
	"glm-4-plus":""
}
base_urls={
	"deepseek-v3":"https://dashscope.aliyuncs.com/compatible-mode/v1",
	"Baichuan4-Turbo":"https://api.baichuan-ai.com/v1",
	"qwen-max":"https://dashscope.aliyuncs.com/compatible-mode/v1",
	"glm-4-plus":"https://open.bigmodel.cn/api/paas/v4"
}
model_name="deepseek-v3"
llm = ChatOpenAI (
	api_key=api_keys.get(model_name),
	base_url=base_urls.get(model_name),
	model=model_name,
	max_tokens = 80000,
	top_p = 0.9,
	temperature = 0.3,
	timeout=100000
)
answer_template = """
帮我回答一下下面的问题。
问题：{question}
答案：
"""
prompt = PromptTemplate.from_template(answer_template)
chain = LLMChain(llm=llm, prompt=prompt)
ques=pd.read_excel("eval_data/ques.xlsx")
questions=[]
ground_truths=[]
contexts=[]
answer=[]
cypher=[]
datas=[]
for i in ques.itertuples() :
	result = chain.invoke({"question": i.question})
	print(result)
	questions.append(i.question)
	ground_truths.append(i.ground_truth)
	contexts.append(i.retrieved_contexts)
	answer.append(result["text"])
data={
	"user_input": questions,
	"response": answer,
}
da=pd.DataFrame.from_dict(data)
file_name=f"res-{model_name}.csv"
da.to_csv(file_name)
	