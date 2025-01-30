#!/usr/bin/env python
# coding: utf-8

# In[71]:


import os
import json
import pandas as pd
import logging
import argparse

from rich import print as pprint
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, ContextEntityRecall, NoiseSensitivity, ResponseRelevancy, Faithfulness, FactualCorrectness, SemanticSimilarity


my_logger = logging.getLogger("my_logger")
my_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
my_logger.addHandler(console_handler)

parser = argparse.ArgumentParser(description="處理指定的文件路徑")
parser.add_argument(
    "file_path",
    type=str,
    help="需指定要評估的CSV檔"
)
args = parser.parse_args()
csv_path = args.file_path


# In[72]:


# user_input 為問題
# reference 為標準答案
# response 為RAG應用所提供的回答
# retrieved_contexts 為RAG所檢索的文檔

file_name = os.path.splitext(os.path.basename(csv_path))[0]       # 提取檔名（不包含路徑和副檔名）
method_name = file_name.replace("eval_", "")                      # 移除 "eval_" 的部分

dataset_df = pd.read_csv(csv_path)
dataset_df["retrieved_contexts"] = dataset_df["retrieved_contexts"].apply(json.loads)
dataset_df = dataset_df[['user_input', 'reference', 'response', 'retrieved_contexts']]
eval_dataset = EvaluationDataset.from_pandas(dataset_df)


# In[74]:


os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model=os.environ["AZURE_OPENAI_CHAT_MODEL_NAME"],
    validate_base_url=False,
))

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    openai_api_version=os.environ["AZURE_OPENAI_API_EMBED_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"],
    model=os.environ["AZURE_OPENAI_EMBED_MODEL_NAME"],
))


# In[75]:


my_logger.info(f"單一文件評估中...")
metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    ContextEntityRecall(llm=evaluator_llm),
    # NoiseSensitivity(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    Faithfulness(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm), 
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=eval_dataset, metrics=metrics)
df_result = results.to_pandas()


# In[78]:


df_result["retrieved_contexts"] = df_result["retrieved_contexts"].apply(json.dumps)
df_result.to_csv(f"./result/result_{method_name}.csv", index=False)

