We use llama-2-7b.Q4_K_M as teacher to generate teacher's responses for the given questions. These responses are stored in jsonl files `fiqa_test_llama-2-7b.Q4_K_M.jsonl` and `Fiq_train_llama_27b.Q4_K_M.jsonl`. Each entry in these files is a dictionary containing:

-"question"
-"ground_truth": ground truth response
-"output_answer": teacher response
-"top_5_indices": top 10 indices from vocabulary for all generated tokens (naming-- top-5 is inaccurate)
-"top_5_logits": top 10 logits from vocabulary for all generated tokens (naming-- top-5 is inaccurate)


| MODEL  |Bert_Score_f1 |Bert_Score_precision |Bert_Score_recall |
|----|----|----|---|
| Teacher (Llama2-7B-Q4) | 73.93 | 73.50 | 74.52 |
