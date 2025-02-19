from transformers import AutoTokenizer
import argparse
import json
from datasets import Dataset
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.model_utils import init_inference_model
from src.inference_utils import generate
from src.data_utils import init_dataset
from src.train_utils import REWARDS

def main(config):
    model = init_inference_model(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    gen_config = model.generation_config

    base_dataset = init_dataset(config["base_dataset"])
    base_dataset = base_dataset["train"].select(list(range(config["inference"]["n_first_samples"])))
    
    raw_dataset = {"prompt":[]}
    for row in base_dataset:
        for _ in range(config["inference"]["n_completions_per_sample"]):
            raw_dataset["prompt"].append(row["prompt"])
    raw_dataset = Dataset.from_dict(raw_dataset)

    sft_dataset = {"prompt":[], "completion":[]}
    for func in config["rejection_sampling"]["reward_funcs"]:
        sft_dataset[func] = []
    for row in raw_dataset:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            completion = generate(model, tokenizer, gen_config, row["prompt"])
        rewards = {}
        for func in config["rejection_sampling"]["reward_funcs"]:
            rewards[func] = REWARDS[config["rejection_sampling"]["reward_group_ind"]]["functions"][func](prompts = row["prompt"], completions = completion)

        sft_dataset["prompt"].append(row["prompt"])
        sft_dataset["completion"].append(completion)
        for key in rewards.keys():
            sft_dataset[key].append(rewards[key])
        
        with open(config["inference"]["path"], "w+") as f:
            json.dump(sft_dataset, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    main(config)