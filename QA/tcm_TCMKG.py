import warnings,ujson
warnings.filterwarnings("ignore")
import pandas as pd
from langchain.chains import GraphCypherQAChain, LLMChain, SequentialChain, TransformChain
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from ragas import EvaluationDataset
import os

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = ""
graph = Neo4jGraph ()
llm = ChatOpenAI (
	temperature=0,
	openai_api_key="openai_api_key",
	openai_api_base="base_url",
	model="gpt-3.5-turbo",
)
def execute_cypher (inputs: dict) -> dict :
	cypher_query = inputs ["cypher"]
	try :
		result = graph.query ( cypher_query )
		return {"data" : str ( [dict ( item ) for item in result] )}
	except Exception as e :
		return {"data" : f"查询错误：{str ( e )}"}

description_query = """
MATCH (m:疾病|症状|病因病机|脉象|方剂|中药|舌象|治法治则)
WHERE m.name="candidate"
MATCH (m)-[r:呈现|引发|关联|适配|遵循|对应|产生|应用|组成|推断]-(t)
WITH m, type(r) as type, collect(coalesce(t.name)) as names
WITH m, type+": "+reduce(s="", n IN names | s + n + ", ") as types
WITH m, collect(types) as contexts
WITH m, "name:" + labels(m)[0] + "\nmessages: "+ coalesce( m.name)
+ "\nname: "+coalesce(m.name,"") +"\n" +
reduce(s="", c in contexts | s + substring(c, 0, size(c)-2) +"\n") as context
RETURN context LIMIT 1
"""

cypher_template = """
你是一个Neo4j专家，能将问题中抽取出相关实体插入到Cypher模板中进行查询。
关系只包含以下的关系：
呈现、引发、关联、适配、遵循、对应、产生、应用、组成、推断
数据库Schema：
{schema}
Cypher模板：
{Cypher_template}
问题：{question}
输出为Cypher语句，无其他不相关的文本例如：cypher和字符如"```"。
请生成Cypher查询：
"""
answer_template = """
帮我根据问题和数据，给出回答，无需给出分析。
问题：{question}
数据：{data}
答案：
"""
cypher_prompt = PromptTemplate.from_template(cypher_template)
cypher_chain = LLMChain(llm=llm, prompt=cypher_prompt, output_key="cypher")
query_execution_chain = TransformChain(
	input_variables=["cypher"],
	output_variables=["data"],
	transform=execute_cypher
)
answer_prompt = PromptTemplate.from_template(answer_template)

answer_chain = LLMChain(llm=llm, prompt=answer_prompt, output_key="answer")
overall_chain = SequentialChain(
	chains=[cypher_chain, query_execution_chain, answer_chain],
	input_variables=["question", "schema","Cypher_template"],
	output_variables=["cypher", "data", "answer"],
)
result = overall_chain.invoke({"question": "你好", "schema": graph.get_schema,"Cypher_template":description_query})
print(result["answer"])