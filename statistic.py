#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import pandas as pd
import logging
import argparse


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


# In[9]:

df_result = pd.read_csv(csv_path)
df_result["retrieved_contexts"] = df_result["retrieved_contexts"].apply(json.loads)
selected_columns = ["context_recall", "context_entity_recall", "answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"]
column_means = df_result[selected_columns].mean()
column_means["user_input"] = "mean"
column_means["retrieved_contexts"] = []
column_means["response"] = "mean"
column_means["reference"] = "mean"
df_result.loc["mean"] = column_means
df_result["retrieved_contexts"] = df_result["retrieved_contexts"].apply(json.dumps)
df_result.to_csv(f"{csv_path}", index=False)
my_logger.info(f"{csv_path}")

