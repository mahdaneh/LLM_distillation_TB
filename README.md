## Teacher Model and Generated Data

We use `llama-2-7b.Q4_K_M` as the teacher model to generate responses for the given questions.

The generated outputs are stored in the following JSONL files:

- `fiqa_test_llama-2-7b.Q4_K_M.jsonl`
- `Fiq_train_llama_27b.Q4_K_M.jsonl`

Each entry in these files contains:

- `"question"` – Input question  
- `"ground_truth"` – Reference (gold) answer  
- `"output_answer"` – Teacher-generated response  
- `"top_10_indices"` – Top-10 vocabulary indices for each generated token  
- `"top_10_logits"` – Top-10 logits corresponding to the selected vocabulary indices  

> **Note:** The previous naming (`top_5_indices`, `top_5_logits`) was inaccurate. Each entry stores top-10 values.

---

## Teacher Model Performance

| Model                  | BERTScore F1 | BERTScore Precision | BERTScore Recall |
|------------------------|--------------|---------------------|------------------|
| Teacher (Llama2-7B-Q4) | 73.93        | 73.50               | 74.52            |

The reported scores are computed using BERTScore on the training split.
