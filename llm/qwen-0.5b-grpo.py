"""
Reference:

https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer
from peft import LoraConfig

# load and prepare ds
SYSTEM_PROMPT = """
Responde in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text:str):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text:str):
    if '####' not in text:
        return None
    return text.split('####')[1].strip()

def get_gsm8k_questions(split='train'):
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt':[
            {'role':'system', 'content': SYSTEM_PROMPT},
            {'role':'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data
# step = 0
# reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content'] # [{role:system},{role:user},{role:assistance}]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    if '</answer>' in responses[0] and '<answer>' in responses[0] and '<reasoning>' in responses[0] and '</reasoning>' in responses[0]:
        # print('-'*20, f"Question:\n{q}\n", '-'*20, f"Answer:\n{answer[0]}\n", '-'*20, f"Response:\n{responses[0]}\n", '-'*20, f"Extracted:\n{extracted_responses[0]}",'\n\n')
        print('-'*20, f"Question:\n{q}\n", f"Answer: {answer[0]}\n", f"Extracted: {extracted_responses[0]}")
    return [10.0 if r==a else 0.0 for r,a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs):
    """
    Reward function that checks if the completion has as specific format
    """
    pattern = r"^<reasoning>.*</reasoning>\n<answer>.*?</answer>$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format
    """
    pattern = r"<reasoning>.*</reasoning>\n<answer>.*?</answer>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count+=0.125
    if text.count("\n</reasoning>\n") == 1:
        count+=0.125
    if text.count("\n<answer>\n") == 1:
        count+=0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count+=0.125
        count-=(len(text.split("\n</answer>")[-1])-1)*0.001
    return count

def xml_count_reward_func(completions, **kwargs):
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig

def train():
    param_size = "0.5B"
    batch_size = 1
    model_name = f"Qwen/Qwen2.5-{param_size}-Instruct"
    
    output_dir=f"outputs/Qwen-{param_size}-GRPO"
    run_name=f"QWEN-{param_size}-GRPO-gsm8k-1"
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.3,
        vllm_device='cuda:0',
        report_to='wandb',
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------>start train------------------>

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type='CAUSAL_LM'
    )
    
    
    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[
            xml_count_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        # peft_config
    )
    
    print('param=', param_size)
    trainer.train()
    
    trainer.save_model(output_dir)


