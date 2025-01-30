#!/bin/bash

root_path="/c/NTUST/Research"
final_research_parent_folder=$root_path"/research_result"

for dir in $final_research_parent_folder/*/; do
  business_name=$(basename "$dir")
  bash run_evaluate.sh $business_name
done