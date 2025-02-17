from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import wandb
from huggingface_hub import login
from peft import LoraConfig
import torch
from accelerate import notebook_launcher, Accelerator
import argparse
import json

from src.data_utils import init_dataset
from src.train_utils import init_model, init_trainer

def main(config):
    dataset = init_dataset(config["dataset"])
    model = init_model(config["model"])
    trainer = init_trainer(config["trainer"], dataset, model)
    if "resume_from_checkpoint" in config["trainer"]:
        trainer.train(resume_from_checkpoint=config["trainer"]["resume_from_checkpoint"])
    else:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    main(config)