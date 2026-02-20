from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utility import generat_text as gen
from tqdm import tqdm
from dataset_prepare import *
from pathlib import Path
# from Evaluate_student import *
device = "cuda"
student_name = ""
# student_name = "./DistilLlama_Fiq_train_llama_27b.Q4_K_M_llama-2-7b.Q4_K_M_distilled"
dataset_name ="./Fiq_train_llama_27b.Q4_K_M.jsonl"
test_dataset_name = "./fiqa_test_llama-2-7b.Q4_K_M_test.jsonl"
teacher_model_name = "llama-2-7b.Q4_K_M.gguf"
teacher_results_fn =f"{dataset_name}_{teacher_model_name}_results.json"
tokenizer = AutoTokenizer.from_pretrained("./llama-7b")
student = AutoModelForCausalLM.from_pretrained(student_name).to(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)


train_dataset = JsonlDataset(dataset_name, tokenizer, max_length=250)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Hyperparameters
batch_size = 4
num_epochs = 1
learning_rate = 5e-5
alpha = 0.5  # weight for teacher imitation loss vs GT

loss_fct = CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
            optimizer.zero_grad()

            # -------------------------
            # Teacher imitation loss
            # -------------------------
            input_ids_imi = batch["input_ids_imi"].to(device)
            attention_mask_imi = batch["attention_mask_imi"].to(device)
            labels_imi = batch["labels_imi"].to(device)

            outputs_imi = student(input_ids=input_ids_imi, attention_mask=attention_mask_imi)

            logits_imi = outputs_imi.logits  # [batch, seq_len, vocab_size]

            loss_imi = loss_fct(logits_imi.view(-1, logits_imi.size(-1)), labels_imi.view(-1))

            # -------------------------
            # Ground-truth loss
            # -------------------------
            input_ids_gt = batch["input_ids_gt"].to(device)
            attention_mask_gt = batch["attention_mask_gt"].to(device)
            labels_gt = batch["labels_gt"].to(device)

            outputs_gt = student(input_ids=input_ids_gt, attention_mask=attention_mask_gt)



            logits_gt = outputs_gt.logits

            loss_gt = loss_fct(logits_gt.view(-1, logits_gt.size(-1)), labels_gt.view(-1))
            # # -------------------------
            # # KL Divergence on teacher's top-k logits
            # # -------------------------
            # temperature=2
            # teacher_top_logits = batch["teacher_top_logits"].to(device)  # [B, L, K]
            # teacher_top_indices = batch["teacher_top_indices"].to(device)  # [B, L, K]
            # student_logits_flat = outputs_imi.view(-1, logits_imi.size(-1))  # [B*L, vocab_size]
            # labels_flat = loss_imi.view(-1)  # [B*L]
            #
            # # mask positions
            # mask_flat = labels_flat != -100
            # student_logits_teacher_tokens = student_logits_flat[mask_flat]  # [num_teacher_tokens, vocab_size]
            # teacher_labels_tokens = labels_flat[mask_flat]  # [num_teacher_tokens]
            # # teacher_probs: [B,L,K]
            # teacher_probs = F.softmax(teacher_top_logits / temperature, dim=-1)
            #
            # # student logits at teacher indices
            # student_top_logits = torch.gather(outputs_imi, 2, teacher_top_indices)
            #
            # # log softmax over same top-k indices
            # student_log_probs = F.log_softmax(student_top_logits / temperature, dim=-1)
            #
            # # KL
            # kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            # -------------------------
            # Combined loss
            # -------------------------
            loss = alpha * loss_imi + (1 - alpha) * loss_gt

            # Backpropagation
            loss.backward()
            optimizer.step()




            loop.set_postfix(loss=loss.item())

    # student.eval()
    # eval_rs = evaluate_student(student, tokenizer, test_dataset_name, batch_size=4)
    # print(eval_rs)
    # exit()
student.save_pretrained(f"./{Path(student_name).stem}_{Path(dataset_name).stem}_{Path(teacher_model_name).stem}_distilled")

