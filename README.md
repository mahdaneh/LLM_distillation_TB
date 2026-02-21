## Teacher Model and Generated Data

We use `llama-2-7b.Q4_K_M` as the teacher model to generate offline responses for the given questions.

The generated offline responses are stored in the following JSONL files that could be used for training and evaluating the student model:

- `fiqa_test_llama-2-7b.Q4_K_M.jsonl`
- `Fiq_train_llama_27b.Q4_K_M.jsonl`

Each entry in these files contains:

- `"question"` – Input question  
- `"ground_truth"` – Reference (gold) answer  
- `"output_answer"` – Teacher-generated response  
- `"top_5_indices"` – Top-10 vocabulary indices for each generated token  
- `"top_5_logits"` – Top-10 logits corresponding to the selected vocabulary indices  



---

## Teacher Model Performance

As BERTScore compares contextual embeddings of tokens using a pretrained transformer, it is more suitable for evaluating LLM, to assess accuracy and relatedness of the generated response (w.r.t gt response) 

| Model                  | BERTScore F1 | BERTScore Precision | BERTScore Recall |
|------------------------|--------------|---------------------|------------------|
| Teacher (Llama2-7B-Q4) | 73.93        | 73.50               | 74.52            |

The reported scores are computed using BERTScore on the training split.
