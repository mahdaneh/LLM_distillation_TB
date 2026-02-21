from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mlflow
import torch
import numpy as np
import evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
bertscore_metric = evaluate.load("bertscore")
bertscore_metric.device = "cuda"
import json
class QAEvalDataset(Dataset):
    def __init__(self, file_path):
        # Load JSONL data
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "ground_truth": item["answer"],
        }
def collate_fn(batch):
    questions = [x["question"] for x in batch]
    gts = [x["ground_truth"] for x in batch]
    return questions, gts

def generate_student_batch(model, tokenizer, questions, max_new_tokens=128):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    # import pdb; pdb.set_trace()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(
        outputs[0],
        skip_special_tokens=True
    )

    return decoded

from tqdm import tqdm
def evaluate_student(model, tokenizer, data_filename, batch_size=1):

    dataset = QAEvalDataset(data_filename)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


    bert_scores = {"precision": [], "recall": [], "f1": []}

    sample_offset = 0

    with mlflow.start_run(nested=True):

        for questions, refs in tqdm(loader):

            preds = generate_student_batch(model, tokenizer, questions)


            bs = bertscore_metric.compute(
                predictions=preds,
                references=refs,
                lang="en",
                model_type="distilbert-base-uncased"
            )

            bert_scores["precision"].extend(bs["precision"])
            bert_scores["recall"].extend(bs["recall"])
            bert_scores["f1"].extend(bs["f1"])

            sample_offset += len(preds)


            results = {

                "BERTScore_f1": np.mean(bert_scores["f1"]),
                "BERTScore_precision": np.mean(bert_scores["precision"]),
                "BERTScore_recall": np.mean(bert_scores["recall"]),
            }
            print(results)

        mlflow.log_metrics(results)

    return results

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    model_name = "./student_llma_Fiq_train_llama_27b.Q4_K_M_llama-2-7b.Q4_K_M_qlora"
    data_filename = "./fiqa_test.json"
    tokenizer = AutoTokenizer.from_pretrained("./llama-7b")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    evaluate_student(model,tokenizer, data_filename)
