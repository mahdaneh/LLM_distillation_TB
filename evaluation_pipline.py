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
bertscore_metric = evaluate.load("bertscore")
bertscore_metric.device = "cuda"


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



    bert_scores = {"precision": [], "recall": [], "f1": []}






    with mlflow.start_run(run_name=mlflow_run, nested=True):

        chunk_size = 500

        for start in range(0, len(preds), chunk_size):
            end = start + chunk_size
            preds_chunk = preds[start:end]
            refs_chunk = gts[start:end]

            # BERTScore

            bs = bertscore_metric.compute(predictions=preds_chunk,
                                     references=refs_chunk, lang="en",model_type = "distilbert-base-uncased")
            print(bs)
            bert_scores["precision"].extend(bs["precision"])
            bert_scores["recall"].extend(bs["recall"])
            bert_scores["f1"].extend(bs["f1"])

            # meteor_scores.append(meteor_metric.compute(predictions=preds_chunk,references=refs_chunk)["meteor"])






            results = {

                 "BERTScore_f1": np.mean(bert_scores["f1"]),
                "BERTScore_precision": np.mean(bert_scores["precision"]),
                "BERTScore_recall": np.mean(bert_scores["recall"]),
            }
            print(results)
    mlflow.log_metrics(results)

    return results


if __name__ == "__main__":
    print("Evaluation Results:")
    evaluate("fiqa_test_llama-2-7b.Q4_K_M_test.jsonl",
             mlflow_experiment="LLM_Evaluation",
             mlflow_run=f"Fiq_train_llama_27b.Q4_K_M_Evaluation")

