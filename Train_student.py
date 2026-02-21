from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataset_prepare import *
from pathlib import Path
import trackio
device = "cuda"


batch_size = 6
num_epochs = 5
learning_rate = 2e-4
alpha = 0.5
gradient_accumulation_steps = 16

student_name = "./student_llma"
dataset_name ="./Fiq_train_llama_27b.Q4_K_M.jsonl"
teacher_model_name = "llama-2-7b.Q4_K_M.gguf"

# ========================
# 4-bit Quantization Config
# ========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained(student_name)
tokenizer.pad_token = tokenizer.eos_token


student = AutoModelForCausalLM.from_pretrained(
    student_name,
    quantization_config=bnb_config,
    device_map="auto",
)

print("=== Loaded 4-bit student model ===")


student.gradient_checkpointing_enable()
student.config.use_cache = False


student = prepare_model_for_kbit_training(student)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # For LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

student = get_peft_model(student, lora_config)

student.print_trainable_parameters()


optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)


train_dataset = JsonlDataset(dataset_name, tokenizer, max_length=250)
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

loss_fct = CrossEntropyLoss(ignore_index=-100)

# ====== Trackio
config ={
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "grad_accum_steps": gradient_accumulation_steps,
    "alpha": alpha,
}
trackio.init(project="student-distillation", config=config)


student.train()
global_step = 0
for epoch in range(num_epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    epoch_loss = 0
    epoch_steps = 0
    for step, batch in enumerate(loop):
        if batch is None:
            continue


        # Teacher imitation loss

        input_ids_imi = batch["input_ids_imi"].to(device)
        attention_mask_imi = batch["attention_mask_imi"].to(device)
        labels_imi = batch["labels_imi"].to(device)

        outputs_imi = student(
            input_ids=input_ids_imi,
            attention_mask=attention_mask_imi
        )

        logits_imi = outputs_imi.logits

        loss_imi = loss_fct(
            logits_imi.view(-1, logits_imi.size(-1)),
            labels_imi.view(-1)
        )


        # Ground-truth loss

        input_ids_gt = batch["input_ids_gt"].to(device)
        attention_mask_gt = batch["attention_mask_gt"].to(device)
        labels_gt = batch["labels_gt"].to(device)

        outputs_gt = student(
            input_ids=input_ids_gt,
            attention_mask=attention_mask_gt
        )

        logits_gt = outputs_gt.logits

        loss_gt = loss_fct(
            logits_gt.view(-1, logits_gt.size(-1)),
            labels_gt.view(-1)
        )


        # Combined loss

        loss = alpha * loss_imi + (1 - alpha) * loss_gt

        # Normalize for gradient accumulation
        loss = loss / gradient_accumulation_steps

        loss.backward()

        # Update every 16 steps
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            step_loss = loss.item() * gradient_accumulation_steps

            trackio.log({
                "train/step_loss": step_loss,
                "train/step": global_step,
                "epoch": epoch + 1,
            })

            loop.set_postfix(loss=step_loss)
            epoch_loss += step_loss
            epoch_steps += 1
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        trackio.log({
            "train/epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1,
        })

student = student.merge_and_unload()
path = f"./{Path(student_name).stem}_{Path(dataset_name).stem}_{Path(teacher_model_name).stem}_qlora_{epoch}"
student.save_pretrained(path)
tokenizer.save_pretrained(path)
print(f"Saved merged 4bit model to {path}")

