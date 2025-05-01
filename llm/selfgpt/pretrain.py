import torch
import torch.nn.functional as F
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer
import buddygpt
from buddygpt import GPTConfig, BuddyGPT

output_dir = f'outputs/buddygpt'
tokenizer = AutoTokenizer.from_pretrained('gpt2' ,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
config = GPTConfig(n_block=1024, n_embed=1024, n_head=32, n_layer=16, n_vocab=len(tokenizer), n_kv_head=8)
model = BuddyGPT(config)
model


def print_parameters(model):
    num_param = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'total param {num_param/1024/1024}m')
    
def sample(model, query, max_length=50):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
    )
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen_text

model
print_parameters(model)

from datasets import load_dataset, concatenate_datasets
# 50m model need 20*50m = 1B token
# 100m model need 20*100m = 2B token
# 200m model need 20*200m = 4B token
# 500m model need 20*500m = 10B token

# Total tokens: 1872137976
# 1.8B token
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
# 10B token * 10% = 1B token
web_ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train[:10%]")

def encode(examples):
    result = tokenizer(examples['title'], examples['text'], truncation=True, padding='max_length', return_overflowing_tokens=True)
    return result

def encode2(examples):
    result = tokenizer(examples['text'], truncation=True, padding='max_length', return_overflowing_tokens=True)
    return result

ds = ds.map(encode, batched=True, remove_columns=['url', 'id', 'text', 'title'])
web_ds = web_ds.map(encode2, batched=True, remove_columns=['url','id','text','dump','file_path','language','language_score','token_count','score','int_score'])
ds = concatenate_datasets([ds, web_ds])
ds

# ds['input_ids']

# Load the "all" subset or a specific subject like "computer_science"
cmmlu = load_dataset("haonan-li/cmmlu", "high_school_geography", split='dev')

# We'll use the validation set
# eval_ds = cmmlu["validation"]
def preprocess(example):
    question = example["Question"]
    choices = example["A"], example["B"], example["C"], example["D"]
    context = f"{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n答案是:"

    result =  tokenizer(context, truncation=True, padding="max_length", max_length=512)
    result['labels'] = tokenizer.encode(example['Answer'])
    return result

eval_ds = cmmlu.map(preprocess)
print(eval_ds[0])

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # print(labels)
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

'''
accelerate launch --config_file ptrain.yaml --num_processes=1 pretrain.py
'''
def main():
    from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling
    from datetime import datetime

    # print(sample(model, '中国首都是哪?'))
    FLASH = 1
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_dir = 'outputs/buddygpt'
    class SampleTextCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % 500 == 0:
                prompt = "中国首都是哪?"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids=input_ids,
                    max_length=128,
                )
                gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"\n[Sample generated at step {state.global_step}]:\n{gen_text}\n")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    # TL;DR
    # Action	Why
    # ✅ max_grad_norm=1.0	Clip exploding gradients
    # ✅ Lower learning_rate	Reduce gradient magnitude
    # ✅ Increase warmup_steps	Stabilize early training
    # ✅ Use gradient_accumulation_steps	Smooth out spikes
    # ✅ Monitor layers with high grad norm	Find root cause
    
    args = TrainingArguments(
        run_name=f'buddygpt-{now}',
        output_dir=output_dir,
        learning_rate=2e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=10000,
        bf16=True,
        # fp16=True,
        # max_steps=200,
        # remove_unused_columns=False,
        max_grad_norm=1.0,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        eval_strategy="steps",  # or eval_strategy="steps" in newer versions
        eval_steps=500,              # Correct parameter name
        save_safetensors=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        callbacks=[SampleTextCallback],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    trainer.push_to_hub('learn2pro/buddygpt')

if __name__ == "__main__":
    main()