from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import wandb
from huggingface_hub import login
from peft import LoraConfig
import torch

from src.data_utils import generate_dataset, get_prompt
from src.train_utils import (
    common_format_reward,
    answer_format_reward,
    walls_consistency_reward,
    doors_consistency_reward,
    geometry_consistency_reward,
    prompt_consistency_reward,
    LLMSampleCB
)

def main():
    login()
    #torch.cuda.set_device(0)
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tuned_model_name = "evgmaslov/Qwen2.5-Coder-0.5B-Instruct-flats"
    dataset = generate_dataset()
    dataset = dataset.map(lambda row: {"prompt":get_prompt(row["prompt"], tokenizer)})
    peft_params = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")
    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quant_config,
                                                 device_map="auto",
                                      low_cpu_mem_usage=True, offload_state_dict=True)
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    training_args = GRPOConfig(
        output_dir=tuned_model_name,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=100, 
        report_to="wandb",
        save_total_limit=2,
        push_to_hub=True,
        hub_strategy="checkpoint",
        num_generations=8,
        logging_steps=1,
        temperature=0.9,
        max_prompt_length=2048,
        max_completion_length=2048,
        gradient_accumulation_steps=8)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            common_format_reward,
            answer_format_reward,
            walls_consistency_reward,
            doors_consistency_reward,
            geometry_consistency_reward,
            prompt_consistency_reward,
        ],
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_params,
    )
    wandb_callback = LLMSampleCB(trainer, dataset["test"], num_samples=2, max_new_tokens=8182)
    trainer.add_callback(wandb_callback)
    trainer.train()

if __name__ == "__main__":
    main()