import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import evaluation_pipline as eval_ppl

from llama_cpp import Llama
import os
N_CTX=4096
MAX_NEW_TOKENS=250



def load_model_llama(model_name):
    llm = Llama(model_path=model_name,
                n_threads=os.cpu_count(),
                n_ctx=N_CTX, n_gpu_layers=-1, logits_all=True,
                verbose=False,
                chat_format="llama-2"  )

    return llm

def generate_with_logits_llama(llm_model, query, k=10):

    response = llm_model(

            query,
        max_tokens=MAX_NEW_TOKENS,
        logprobs=10,
        logits_processor=None,  # You can insert custom logic here
        temperature=0.1 , # Keep it low for banking accuracy
    repeat_penalty = 1.2,  # Prevents the model from repeating [INST] tokens
        stream=False

    )



    generated_tokens = response["choices"][0]["logprobs"]["tokens"]
    num_generated_tokens = len(generated_tokens)




    # import pdb; pdb.set_trace()

    logits_array = np.array(llm_model._scores[-num_generated_tokens:,:], copy=False)

    topk_indices = np.argpartition(-logits_array, k, axis=1)[:, :k]
    topk_logits = logits_array[np.arange(len(logits_array))[:, None], topk_indices]

    # top_5_indices = np.argsort(logits_array)[::-1][:,:10]
    # topk_logit_array = logits_array[np.arange(len(logits_array))[:,None],top_5_indices]

    return {
        "Response": response["choices"][0]["text"],
        "top_k_indices": topk_indices,
        "Top_k_Logits": topk_logits
    }


if __name__ == "__main__":


    dataset_name = "./fiqa_test.json"

    model_name="./llama-2-7b.Q4_K_M.gguf"
    # model_name = "./DistilLlama_fiqa_train_llama-2-7b.Q4_K_M_results_1_fixed_llama-2-7b.Q4_K_M_distilled"
    # model_name = "./deepseek-r1-distill-llama-8b.Q4_K_M.gguf"
    # model_name ="./Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf"

    model = load_model_llama(model_name)

    device="cuda" if torch.cuda.is_available() else "cpu"
    data = []
    with open(dataset_name) as f:
        for line in f:
            data.append(json.loads(line))
        f.close()

    from pathlib import Path
    from tqdm import tqdm
    output_filename = f"{Path(dataset_name).stem}_{Path(model_name).stem}_test.jsonl"
    print (f"==== {len(data)} samples ====")
    with open(f"{output_filename}", "a", encoding="utf-8") as f:
        # 8508+228+621

        for i, sample in tqdm(enumerate(data)):




            q = sample["question"]
            gt_a = sample["answer"]


            result = generate_with_logits_llama(model,q,k=10)

            if result is None:
                continue

            final_output = {
                "question": q,
                "ground_truth": gt_a,
                "output_answer": result["Response"],
                "top_5_indices": result["top_k_indices"].tolist(),
                "top_5_logits": result["Top_k_Logits"].tolist()
            }
            f.write(json.dumps(final_output)+"\n")







