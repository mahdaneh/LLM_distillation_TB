from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mlflow
import torch
import numpy as np
import evaluate
squad_metric = evaluate.load("squad")
exact_match_metric = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
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
            "ground_truth": item["ground_truth"],
        }
def collate_fn(batch):
    questions = [x["question"] for x in batch]
    gts = [x["ground_truth"] for x in batch]
    return questions, gts

def generate_student_batch(model, tokenizer, questions, max_new_tokens=128):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        questions,
        # padding=True,
        # truncation=True,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(
        outputs[0],
        # skip_special_tokens=True
    )

    return decoded
def evaluate_student(model, tokenizer, data_filename, batch_size=8):

    dataset = QAEvalDataset(data_filename)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    em_scores = []
    f1_scores = []
    meteor_scores = []
    bert_scores = {"precision": [], "recall": [], "f1": []}

    sample_offset = 0

    with mlflow.start_run(nested=True):

        for questions, refs_chunk in loader:

            preds_chunk = generate_student_batch(model, tokenizer, questions)
            print(preds_chunk)

            # # ---- SQuAD format ----
            # formatted_preds = [
            #     {"id": str(i + sample_offset), "prediction_text": p}
            #     for i, p in enumerate(preds_chunk)
            # ]
            #
            # formatted_refs = [
            #     {"id": str(i + sample_offset),
            #      "answers": {"text": [r], "answer_start": [0]}}
            #     for i, r in enumerate(refs_chunk)
            # ]
            #
            # squad_chunk = squad_metric.compute(
            #     predictions=formatted_preds,
            #     references=formatted_refs
            # )
            #
            # em_scores.append(squad_chunk["exact_match"])
            # f1_scores.append(squad_chunk["f1"])

            # ---- BERTScore ----
            bs = bertscore_metric.compute(
                predictions=preds_chunk,
                references=refs_chunk,
                lang="en",
                model_type="distilbert-base-uncased"
            )

            bert_scores["precision"].extend(bs["precision"])
            bert_scores["recall"].extend(bs["recall"])
            bert_scores["f1"].extend(bs["f1"])

            # # ---- METEOR ----
            # meteor_scores.append(
            #     meteor_metric.compute(
            #         predictions=preds_chunk,
            #         references=refs_chunk
            #     )["meteor"]
            # )

            sample_offset += len(preds_chunk)

        # ---- Final aggregation ----
        results = {
            # "meteor": np.mean(meteor_scores),
            # "F1": np.mean(f1_scores),
            # "EM": np.mean(em_scores),
            "BERTScore_f1": np.mean(bert_scores["f1"]),
            "BERTScore_precision": np.mean(bert_scores["precision"]),
            "BERTScore_recall": np.mean(bert_scores["recall"]),
        }

        mlflow.log_metrics(results)

    return results