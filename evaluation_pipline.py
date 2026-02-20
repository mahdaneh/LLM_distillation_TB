from torch.nn import CrossEntropyLoss
import mlflow
def compute_ppl(model, dataloader):
    model.eval()
    loss_fn = CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += batch["input_ids"].numel()

    return torch.exp(torch.tensor(total_loss / total_tokens))


import re
import string
from collections import Counter
import numpy as np
import torch
import textstat

from rouge_score import rouge_scorer
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utility import *
import evaluate
squad_metric = evaluate.load("squad")
exact_match_metric = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
bertscore_metric.device = "cuda"
# -------------------------
# Main Evaluation Function
# -------------------------

import json
def evaluate(data_filename, mlflow_experiment="LLM_Evaluation", mlflow_run="QA_Evaluation"):
    # data = []
    preds=[]
    gts=[]
    with open(data_filename) as f:
        for line in f:
            line=json.loads(line)
            q = line["question"]
            references = line["ground_truth"]
            predictions = line["output_answer"]

            preds.append(predictions)
            gts.append(references)
            # data.append()
        f.close()
    mlflow.set_experiment(mlflow_experiment)

    print(f"======{len(preds)}")


    em_scores, f1_scores, rouge_scores = [], [], []
    bert_scores = {"precision": [], "recall": [], "f1": []}
    meteor_scores = []

    len_ratios, rep_scores, readability_scores = [], [],[]



    with mlflow.start_run(run_name=mlflow_run, nested=True):

        chunk_size = 500  # safe for CPU + GPU

        for start in range(0, len(preds), chunk_size):
            end = start + chunk_size
            preds_chunk = preds[start:end]
            refs_chunk = gts[start:end]
            # formatted_preds = [{"id": str(i + start), "prediction_text": p} for i, p in enumerate(preds_chunk)]
            # formatted_refs = [{"id": str(i + start), "answers": {"text": [r], "answer_start": [0]}} for i, r in
            #                   enumerate(refs_chunk)]
            # squad_chunk = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
            # em_scores.append(squad_chunk["exact_match"])
            # f1_scores.append(squad_chunk["f1"])
            # f1_scores.append(compute_token_f1(predictions, references))

            # ROUGE
            # rouge_scores.append(rouge_l(predictions, references))

            # BERTScore

            bs = bertscore_metric.compute(predictions=preds_chunk,
                                     references=refs_chunk, lang="en",model_type = "distilbert-base-uncased")
            print(bs)
            bert_scores["precision"].extend(bs["precision"])
            bert_scores["recall"].extend(bs["recall"])
            bert_scores["f1"].extend(bs["f1"])

            meteor_scores.append(meteor_metric.compute(predictions=preds_chunk,references=refs_chunk)["meteor"])






        results = {
            # "meteor": np.mean(meteor_scores),
            # "F1": np.mean(f1_scores),
            # "EM":np.mean(em_scores),
            # "ROUGE-L": np.mean(rouge_scores),
             "BERTScore_f1": np.mean(bert_scores["f1"]),
            "BERTScore_precision": np.mean(bert_scores["precision"]),
            "BERTScore_recall": np.mean(bert_scores["recall"]),
        }
    print(results)
    mlflow.log_metrics(results)

    return results


if __name__ == "__main__":
    print("Evaluation Results:")
    evaluate("Fiq_train_llama_27b.Q4_K_M.jsonl",
             mlflow_experiment="LLM_Evaluation",
             mlflow_run=f"Fiq_train_llama_27b.Q4_K_M_Evaluation")

