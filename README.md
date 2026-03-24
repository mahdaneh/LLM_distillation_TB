
# Distillation of Large Language Models under Resource Constraints

## Goal

The primary goal of this project is to distill a pre-trained LLM model, called a teacher model, to a smaller language model, called a student model, so that it could be deployed on a resource-limited laptop (with small VRAM, e.g. 8GB) while keeping its performance as high as its teacher counterpart.

We focused on **question asnwering task** in the financial context. And our GPU is **NVIDIA GeForce RTX 4060** with 8GB RAM.

More specifically, the objectives are:
- Enable training and deployment within 8GB VRAM constraints
- Explore efficient offline distillation techniques
- Evaluate the trade-off between compression and performance
- Maintain acceptable semantic understanding on a financial QA task (FiQA dataset)

## Method

We used an **offline teacher–student distillation framework** to avoid loading both the teacher and student models at the same time to the GPU. This method is simple and memory efficiennt.

**Teacher model**
- LLaMA2-7B (quantized version: `llama-2-7b.Q4_K_M`).
- Used via `llama-cpp` for efficient inference.
- Teacher responses and their logits (only top 10 logits per token) are stored (in offline mode) that be later are used for training the student.

**Student model**

-**Architecture**: Custom smaller LLaMA-based architecture with 4 transformer layers, Hidden size: 512, 8 attention heads, making up total parameters ~108M.

-**Loss function**: Student is trained using cross-entropy loss (with GT tokens) and KL divergence loss(to align probability distributions to mimic teacher outputs).

-**Training considerations**: QLoRA used for memory efficiency (4-bit quantization) with LoRA rank: 16, batch size: 2 with gradient accumulation.

Note that we used a shared tokenizer between the teacher and the student.

### Dataset
- FiQA (financial question answering)
- 14,511 training samples
- 2,561 test samples (a subset used due to time limits)

The generated offline responses by the teacher are stored in the following JSONL files that could be used for training and evaluating the student model:

- `fiqa_test_llama-2-7b.Q4_K_M.jsonl`
- `Fiq_train_llama_27b.Q4_K_M.jsonl`

Each entry in these files contains:

- `"question"` – Input question  
- `"ground_truth"` – Reference (gold) answer  
- `"output_answer"` – Teacher-generated response  
- `"top_10_indices"` – Top-10 vocabulary indices for each generated token  
- `"top_10_logits"` – Top-10 logits corresponding to the selected vocabulary indices  


### Evaluation
- Metrics: F1, Precision, Recall (BERT-based semantic evaluation)
- Focus on semantic similarity rather than exact match

## Results


The performance of teacher and student models on the FiQA test set is summarized below:

| Model                      | F1    | Precision | Recall |
|---------------------------|-------|----------|--------|
| Teacher (LLaMA2-7B Q4)    | 73.93 | 73.50    | 74.52  |
| Student (Custom LLaMA)    | 64.50 | 62.30    | 67.13  |

Key observations:
- The student model achieves reasonable performance despite significant compression
- There is a noticeable performance gap due to reduced model capacity
- The student retains semantic understanding but with lower accuracy

---

## Discussion

This project demonstrates that **offline distillation is a practical solution under strict hardware constraints**. By precomputing teacher outputs, the need for simultaneous model loading is eliminated, enabling training on low-memory systems.

**Limitations:**
- **Model Compression vs Performance**: Reducing the model to 4 layers and ~108M parameters leads to performance degradation
- **Response-Based Limitation**: While efficient, it fails to transfer uncertainty and full probability distributions
- **Logit Storage Trade-off**: Storing only top-10 logits reduces memory usage but loses part of the information
- **Quantization Impact**: 4-bit quantization and QLoRA enable training but may slightly reduce accuracy

Despite these limitations, the student model maintains **acceptable semantic performance**, making it suitable for deployment in resource-constrained environments.

Future improvements could include:
- Gradual model size reduction instead of aggressive compression
- Full implementation of logit-based distillation
- Exploration of attention or feature-based distillation methods
- Training on larger subsets of the dataset

Overall, this work highlights a **practical pathway for deploying LLMs in low-resource settings** while balancing efficiency and performance.
