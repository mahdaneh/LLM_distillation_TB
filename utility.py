import torch.nn.functional as F
import re
import string
from collections import Counter
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL divergence loss
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def normalize_text(text):
    """Lowercase, remove punctuation and extra spaces."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join(text.split())
    return text


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

from bert_score import score
def bert_score_metric(preds, gts, lang="en"):
    P, R, F1 = score(preds, gts, lang=lang, verbose=False)
    return F1.mean().item()

from rouge_score import rouge_scorer
# ROUGE-L scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l(pred, gt):
    scores = scorer.score(gt, pred)
    return scores["rougeL"].fmeasure
def compute_token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
import torch
def generat_text(model,tokenizer,questions,max_new_tokens=None):
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
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return decoded