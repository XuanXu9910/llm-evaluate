#!/bin/bash

# 使用者輸入ESG報告書檔名
report_name=$1

root_path="/c/NTUST/Research"
final_research_folder=$root_path"/research_result/"$report_name
evaluate_folder="./eval"
result_folder="./result"

echo "-----------------開始進行評估: $report_name-----------------"

# 遍歷來源資料夾中的所有檔案，篩選以 eval 開頭且副檔名為 .csv 的檔案
for file_path in $final_research_folder/eval_community*.csv; do
  # 檢查檔案是否存在，避免空匹配
  if [ -f $file_path ]; then
    cp $file_path $evaluate_folder
    eval_with_extension=$(basename $file_path)
    eval_without_extension=${eval_with_extension%.*}
    eval_name="${eval_without_extension#*_}"
    result_path=$result_folder/result_$eval_name.csv
    python evaluateLLM.py $file_path
    python statistic.py $result_path
    rm -r $evaluate_folder
    mkdir $evaluate_folder
    mv $result_path $final_research_folder
  else
    echo "找不到符合條件的檔案！"
  fi
done



echo "-----------------完成評估: $report_name-----------------"