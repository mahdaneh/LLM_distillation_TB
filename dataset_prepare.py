import json
import torch
from torch.utils.data import Dataset, DataLoader


class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    def tokenize_input(self,text):
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Flatten tensors
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            prompt_ids, _ = self.tokenize_input(item["question"])
            response_ids, _ = self.tokenize_input(item["ground_truth"])
            # Skip empty tensors
            if prompt_ids.ndim == 0 or response_ids.ndim == 0:
                return None
            if len(prompt_ids) == 0 or len(response_ids) == 0:
                return None


            input_ids_gt = torch.cat([prompt_ids, response_ids], dim=0)
            attention_mask_gt = torch.ones_like(input_ids_gt)
            labels_gt = input_ids_gt.clone()
            labels_gt[:len(prompt_ids)] = -100  # mask prompt


            teacher_response_ids, _ = self.tokenize_input(item["output_answer"])
            input_ids_imi = torch.cat([prompt_ids, teacher_response_ids], dim=0)
            attention_mask_imi = torch.ones_like(input_ids_imi)
            labels_imi = input_ids_imi.clone()
            labels_imi[:len(prompt_ids)] = -100  # mask prompt

            teacher_top_logits = torch.tensor(item["top_5_logits"], dtype=torch.float)  # [seq_len, top_k]
            teacher_top_indices = torch.tensor(item["top_5_indices"], dtype=torch.long)  # [seq_len, top_k]
            if teacher_top_logits.ndim < 2 or teacher_top_indices.ndim < 2:
                return None

            return {
                "input_ids_gt": input_ids_gt,
                "attention_mask_gt": attention_mask_gt,
                "labels_gt": labels_gt,
                "input_ids_imi": input_ids_imi,
                "attention_mask_imi": attention_mask_imi,
                "labels_imi": labels_imi,
                "teacher_top_logits": teacher_top_logits,
                "teacher_top_indices": teacher_top_indices,
                "question":item["question"],
                "gt":item["ground_truth"],
                "teacher_response":item["output_answer"]

            }
        except Exception:
            # If anything unexpected happens, skip sample
            return None






from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    # If entire batch is invalid,skip
    if len(batch) == 0:
        return None

    # #  remove samples with zero-length tensors
    # valid_batch = []
    # for b in batch:
    #     if (
    #             b["input_ids_gt"].numel() == 0 or
    #             b["input_ids_imi"].numel() == 0 or
    #             b["teacher_top_logits"].numel() == 0 or
    #             b["teacher_top_indices"].numel() == 0
    #     ):
    #         continue
    #     valid_batch.append(b)
    #
    # if len(valid_batch) == 0:
    #     return None
    #
    # batch = valid_batch
    # Ground truth
    input_ids_gt = [b['input_ids_gt'] for b in batch]
    attention_mask_gt = [b['attention_mask_gt'] for b in batch]
    labels_gt = [b['labels_gt'] for b in batch]

    # Imitation (teacher)
    input_ids_imi = [b['input_ids_imi'] for b in batch]
    attention_mask_imi = [b['attention_mask_imi'] for b in batch]
    labels_imi = [b['labels_imi'] for b in batch]

    # Pad sequences
    input_ids_gt = pad_sequence(input_ids_gt, batch_first=True, padding_value=0)
    attention_mask_gt = pad_sequence(attention_mask_gt, batch_first=True, padding_value=0)
    labels_gt = pad_sequence(labels_gt, batch_first=True, padding_value=-100)

    input_ids_imi = pad_sequence(input_ids_imi, batch_first=True, padding_value=0)
    attention_mask_imi = pad_sequence(attention_mask_imi, batch_first=True, padding_value=0)
    labels_imi = pad_sequence(labels_imi, batch_first=True, padding_value=-100)

    teacher_top_logits = pad_sequence([b['teacher_top_logits'] for b in batch], batch_first=True, padding_value=0.0)
    teacher_top_indices = pad_sequence([b['teacher_top_indices'] for b in batch], batch_first=True, padding_value=0)

    return {
        "input_ids_gt": input_ids_gt,
        "attention_mask_gt": attention_mask_gt,
        "labels_gt": labels_gt,
        "input_ids_imi": input_ids_imi,
        "attention_mask_imi": attention_mask_imi,
        "labels_imi": labels_imi,
        "teacher_top_logits": teacher_top_logits,
        "teacher_top_indices": teacher_top_indices,
        "questions": [b["question"] for b in batch],
        "gts": [b["gt"] for b in batch]
    }