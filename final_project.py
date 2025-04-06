import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = None

# Log-likelihood function
def compute_log_likelihood(prompt, answer, model, tokenizer):
    full_input = prompt + "\nAnswer: " + answer
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tokenizer(prompt + "\nAnswer: ", return_tensors="pt").input_ids.shape[-1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # Mask prompt

    with torch.no_grad():
        loss = model(input_ids, labels=labels).loss.item()
        return -loss * (input_ids.shape[-1] - prompt_len)  # Approximate log-likelihood

# Load prompts
df = pd.read_csv("bbq_prompts_no_cot.csv")

correct = 0
total = 0
results = []

for i, row in df.iterrows():
    prompt = row["formatted_prompt"]
    true_label = row["correct_label"]
    choices = eval(row["answer_choices"]) if isinstance(row["answer_choices"], str) else row["answer_choices"]

    scores = []
    for j, choice in enumerate(choices):
        full_answer = f"{j} {choice}"
        score = compute_log_likelihood(prompt, full_answer, model, tokenizer)
        scores.append(score)

    pred = int(torch.tensor(scores).argmax().item())
    results.append({
        "true_label": true_label,
        "llm_label": pred,
        "is_correct": pred == true_label,
        "context_condition": row["context_condition"],
        "prompt": prompt,
        "scores": scores,
        "answer_choices": choices
    })

    if pred == true_label:
        correct += 1
    total += 1

    if i % 10 == 0:
        print(f"Processed {i} examples")

# Save results
print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
df_results = pd.DataFrame(results)
df_results.to_csv("noCOT_bbq_likelihood_eval.csv", index=False)

