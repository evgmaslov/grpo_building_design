import re
import json
from shapely.geometry import Polygon
import random
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from transformers.integrations import WandbCallback
import torch
import wandb
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
import torch
from accelerate import Accelerator

from .inference_utils import generate

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=2, max_new_tokens=2048):
        super().__init__()
        self.sample_dataset = test_dataset.select(random.choices(range(len(test_dataset)), k=num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.inference_model = None
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
        self.gen_config.t = 0.9

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"])
        prompts = examples["prompt"]
        generations = generate(self.inference_model, self.tokenizer, self.gen_config, prompts)
        for i in range(len(prompts)):
            records_table.add_data(prompts[i], generations[i])
        return records_table

    def on_log(self, args, state, control,  **kwargs):
        super().on_log(args, state, control, **kwargs)
        self.inference_model = self.model
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})



def check_format(json, format):
    if type(json) != type(format):
        return 0
    if type(json) == type({}):
        json_keys = set([k for k in json.keys()])
        format_keys = set([k for k in format.keys()])
        keys_intersection = json_keys.intersection(format_keys)
        local_reward = 0
        max_local_reward = max(len(format_keys), len(json_keys))
        
        for key in keys_intersection:
            local_result = check_format(json[key], format[key])
            local_reward += local_result
        local_reward = local_reward / max_local_reward
        return local_reward
    elif type(json) == type([]):
        if len(format) == 0:
            return 1
        base_format = format[0]
        local_reward = 0
        max_local_reward = len(json)
        for element in json:
            local_result = check_format(element, base_format)
            local_reward += local_result
        local_reward = local_reward / max_local_reward if max_local_reward > 0 else 1
        return local_reward
    else:
        return 1

def common_format_reward(**kwargs):
    completions = kwargs["completions"]
    rewards = []
    for c in completions:
        reward = 0.0
        if "<think>" in c:
            reward += 1/8
        if "</think>" in c:
            reward += 1/8
        if "<answer>" in c:
            reward += 1/8
        if "</answer>" in c:
            reward += 1/8
        pattern_1 = "<think>(?:.|\n)*?</think>"
        if len(re.findall(pattern_1, c)) == 1:
            reward += 1/8
        pattern_2 = "<answer>(?:.|\n)*?</answer>"
        if len(re.findall(pattern_2, c)) == 1:
            reward += 1/8
        if "</think><answer>" in c:
            reward += 1/8
        pattern_3 = "<think>(?:.|\n)*?</think><answer>(?:.|\n)*?</answer>"
        if len(re.findall(pattern_3, c)) == 1:
            reward += 1/8
        rewards.append(reward)
    return rewards

def cot_length_reward(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = common_format_reward(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<think>).*?(?=\</think\>)"
            cot = re.findall(pattern, c)[0]
            max_len = 3000
            local_reward = (len(cot) - max_len)/max_len
            rewards.append(len(cot))
        except:
            rewards.append(0)
    return rewards


def answer_format_reward_1(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    for i in range(len(completions)):
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            try:
                answer_json = json.loads(answer)
            except:
                rewards.append(0)
                continue
            
            answer_format = REWARDS[0]["format"]
            result = check_format(answer_json, answer_format)
            rewards.append(result)
        except:
            rewards.append(0)
    return rewards

def walls_consistency_reward_1(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = answer_format_reward_1(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            max_local_reward = len(answer_json["rooms"])
            for room in answer_json["rooms"]:
                prev_point = room["walls"][-1]["end_point"]
                wall_local_reward = 0
                max_wall_local_reward = len(room["walls"])
                for wall in room["walls"]:
                    cur_point = wall["start_point"]
                    if cur_point[0] == prev_point[0] and cur_point[1] == prev_point[1]:
                        wall_local_reward += 1
                    prev_point = wall["end_point"]
                wall_local_reward = wall_local_reward / max_wall_local_reward
                local_reward += wall_local_reward
            local_reward = local_reward / max_local_reward

            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def doors_consistency_reward_1(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = answer_format_reward_1(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            max_local_reward = len(answer_json["rooms"])
            passed_cons = []
            good_cons = []
            for room in answer_json["rooms"]:
                local_cons = []
                failed_walls = False
                for wall in room["walls"]:
                    start = wall["start_point"]
                    end = wall["end_point"]
                    if start[0] != end[0] and start[1] != end[1]:
                        failed_walls = True
                        break
                    coord = start[0] if start[0] == end[0] else start[1]
                    for door in wall["doors_to"]:
                        local_cons.append({
                            "room":door,
                            "coord":coord
                        })
                if failed_walls:
                    continue
                room_reward = 0
                max_room_reward = len(local_cons)
                for con in local_cons:
                    str_con = "@".join(sorted([room["name"], con["room"]]) + [str(con["coord"])])
                    if str_con in good_cons:
                        room_reward += 1
                    if str_con in passed_cons:
                        continue
                    passed_cons.append(str_con)
                    con_room = [r for r in answer_json["rooms"] if r["name"] == con["room"]]
                    if len(con_room) == 0:
                        continue
                    con_room = con_room[0]
                    con_walls = [w for w in con_room["walls"] if (w["start_point"][0] == con["coord"] and w["end_point"][0] == con["coord"]) or (w["start_point"][1] == con["coord"] and w["end_point"][1] == con["coord"])]
                    has_con = False
                    for w in con_walls:
                        if room["name"] in w["doors_to"]:
                            has_con = True
                            break
                    if has_con == True:
                        room_reward += 1
                        good_cons.append(str_con)
                room_reward = room_reward / max_room_reward if max_room_reward > 0 else 1
                
                local_reward += room_reward
            local_reward = local_reward / max_local_reward
            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def geometry_consistency_reward_1(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = walls_consistency_reward_1(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            for a in range(len(answer_json["rooms"])):
                for b in range(a + 1, len(answer_json["rooms"])):
                    room_1 = answer_json["rooms"][a]
                    room_2 = answer_json["rooms"][b]
                    pol_1 = Polygon([w["start_point"] for w in room_1["walls"]])
                    pol_2 = Polygon([w["start_point"] for w in room_2["walls"]])
                    intersection = pol_1.intersection(pol_2).area
                    if intersection == 0:
                        local_reward += 1/(len(answer_json["rooms"])*(len(answer_json["rooms"]) - 1)/2)
            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def prompt_consistency_reward_1(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = doors_consistency_reward_1(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            p = prompts[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            pattern = "(?<=\<\|im_start\|\>user\n).*?(?=\<\|im_end\|\>)"
            requirements = json.loads(re.findall(pattern, p)[0])

            p_rooms = set(requirements["rooms"])
            c_rooms = set([r["name"] for r in answer_json["rooms"]])

            p_cons = set(["@".join(sorted(con)) for con in requirements["connections"]])
            c_cons = []
            for room in answer_json["rooms"]:
                con_rooms = []
                for w in room["walls"]:
                    for d in w["doors_to"]:
                        con_rooms.append(d)
                for con_room in con_rooms:
                    str_con = "@".join(sorted([room["name"], con_room]))
                    if str_con not in c_cons:
                        c_cons.append(str_con)
            c_cons = set(c_cons)

            rooms_intersection = p_rooms.intersection(c_rooms)
            cons_intersection = p_cons.intersection(c_cons)
            
            local_reward = len(rooms_intersection) + len(cons_intersection)
            max_local_reward = max(len(p_rooms), len(c_rooms)) + max(len(p_cons), len(c_cons))
            local_reward = local_reward / max_local_reward
            rewards.append(local_reward)
        except:
            rewards.append(0)

    return rewards



def answer_format_reward_2(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    for i in range(len(completions)):
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            try:
                answer_json = json.loads(answer)
            except:
                rewards.append(0)
                continue
            
            answer_format = REWARDS[1]["format"]
            result = check_format(answer_json, answer_format)
            rewards.append(result)
        except:
            raise
            rewards.append(0)
    return rewards

def walls_orthogonality_reward_2(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = answer_format_reward_2(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            max_local_reward = 0
            for room in answer_json["rooms"]:
                name = room["name"]
                polygon = room["polygon"]
                for i in range(len(polygon)):
                    max_local_reward += 1
                    cur_p = polygon[i]
                    last_p = polygon[i - 1]
                    if cur_p[0] == last_p[0] or cur_p[1] == last_p[1]:
                        local_reward += 1
            local_reward = local_reward / max_local_reward if max_local_reward > 0 else 0
            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def doors_consistency_reward_2(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = walls_orthogonality_reward_2(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            max_local_reward = 0
            all_room_names = [room["name"] for room in answer_json["rooms"]]
            polygons = {room["name"]: room["polygon"] for room in answer_json["rooms"]}
            for door in answer_json["doors"]:
                door_name = door["name"]
                max_local_reward += 4
                room_names = door_name.split(" -> ")
                door_point = door["position"]
                if len(room_names) != 2:
                    continue
                for name in room_names:
                    if name not in all_room_names:
                        continue
                    local_reward += 1
                    polygon = polygons[name]
                    is_valid_point = False
                    for i in range(len(polygon)):
                        last_point = polygon[i - 1]
                        cur_point = polygon[i]
                        equal_ind = 0 if last_point[0] == cur_point[0] else 1
                        bound_ind = 0 if equal_ind == 1 else 1
                        min_bound = min(last_point[bound_ind], cur_point[bound_ind])
                        max_bound = max(last_point[bound_ind], cur_point[bound_ind])
                        if door_point[equal_ind] == cur_point[equal_ind] and door_point[bound_ind] >= min_bound and door_point[bound_ind] <= max_bound:
                            is_valid_point = True
                            break
                    if is_valid_point:
                        local_reward += 1
            local_reward = local_reward / max_local_reward if max_local_reward > 0 else 0
            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def geometry_consistency_reward_2(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = doors_consistency_reward_2(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            local_reward = 0
            for a in range(len(answer_json["rooms"])):
                for b in range(a + 1, len(answer_json["rooms"])):
                    room_1 = answer_json["rooms"][a]
                    room_2 = answer_json["rooms"][b]
                    pol_1 = Polygon(room_1["polygon"])
                    pol_2 = Polygon(room_2["polygon"])
                    intersection = pol_1.intersection(pol_2).area
                    if intersection == 0:
                        local_reward += 1/(len(answer_json["rooms"])*(len(answer_json["rooms"]) - 1)/2)
            rewards.append(local_reward)
        except:
            rewards.append(0)
    return rewards

def prompt_consistency_reward_2(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = doors_consistency_reward_2(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            p = prompts[i]
            pattern = "(?<=\<answer>)(?:.|\n)*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            answer_json = json.loads(answer)

            pattern = "(?<=Requirements: )(?:.|\n)*?(?=\nFlat:)"
            requirements = json.loads(re.findall(pattern, p)[0])

            p_rooms = set(requirements["rooms"])
            c_rooms = set([r["name"] for r in answer_json["rooms"]])

            p_cons = set(["@".join(sorted(con)) for con in requirements["connections"]])
            c_cons = set(["@".join(sorted(door["name"].split(" -> "))) for door in answer_json["doors"]])

            rooms_intersection = p_rooms.intersection(c_rooms)
            cons_intersection = p_cons.intersection(c_cons)
            
            local_reward = len(rooms_intersection) + len(cons_intersection)
            max_local_reward = max(len(p_rooms), len(c_rooms)) + max(len(p_cons), len(c_cons))
            local_reward = local_reward / max_local_reward
            rewards.append(local_reward)
        except:
            rewards.append(0)

    return rewards

REWARDS = [
    {"format":{
            "rooms":[
                {"name":"",
                 "walls":[
                    {"start_point": [0, 0], "end_point": [0, 0], "doors_to":[""]},
                 ]},
            ]
        },
    "functions":{
        "common_format_reward":common_format_reward,
        "answer_format_reward":answer_format_reward_1,
        "walls_consistency_reward":walls_consistency_reward_1,
        "doors_consistency_reward":doors_consistency_reward_1,
        "geometry_consistency_reward":geometry_consistency_reward_1,
        "prompt_consistency_reward":prompt_consistency_reward_1
    }},
    {"format":{
            "rooms":[
                {"name":"",
                 "polygon":[[0, 0]]}
            ],
            "doors":[
                {"name":"",
                "position":[0, 0]}
            ]
        },
    "functions":{
        "common_format_reward":common_format_reward,
        "answer_format_reward":answer_format_reward_2,
        "walls_orthogonality_reward":walls_orthogonality_reward_2,
        "doors_consistency_reward":doors_consistency_reward_2,
        "geometry_consistency_reward":geometry_consistency_reward_2,
        "prompt_consistency_reward":prompt_consistency_reward_2
    }},
]

def init_trainer(config, dataset, model):
    training_args = GRPOConfig(
        output_dir=config.get("output_dir"),
        num_train_epochs=config.get("num_train_epochs"),
        per_device_train_batch_size=config.get("per_device_train_batch_size"),
        save_steps=config.get("save_steps", 100), 
        report_to=config.get("report_to", "wandb"),
        save_total_limit=config.get("save_total_limit", 2),
        push_to_hub=config.get("push_to_hub", True),
        hub_strategy=config.get("hub_strategy", "checkpoint"),
        num_generations=config.get("num_generations", 8),
        logging_steps=config.get("logging_steps", 1),
        temperature=config.get("temperature", 0.9),
        max_prompt_length=config.get("max_prompt_length", 2048),
        max_completion_length=config.get("max_completion_length", 2048),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("learning_rate", 8e-6))
    reward_funcs = [REWARDS[config["reward_group_ind"]]["functions"][name] for name in config["reward_funcs"]]
    peft_config = None
    if "peft_config" in config:
        peft_config = LoraConfig(
            lora_alpha=16, 
            lora_dropout=0.1, 
            r=64, bias="none", 
            task_type="CAUSAL_LM",
            target_modules=config["peft_config"].get("target_modules", None))
    trainer_type = config.get("trainer_type", GRPOTrainer)
    if trainer_type == "per_token_grpo":
        trainer_type = TGRPOTrainer
    elif trainer_type == "fixed_grpo":
        trainer_type = FixedGRPOTrainer
    trainer = trainer_type(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
    )
    
    return trainer


from typing import Any, Callable, Optional, Sized, Union
from contextlib import contextmanager
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import torch.nn.functional as F
from tqdm.auto import tqdm

@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    is_peft_model: bool = False,
    gather_deepspeed3_params: bool = True,
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        raise NotImplementedError
    else:
        yield unwrapped_model

class TGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = prompts
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        #device_index = Accelerator().process_index
        #prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(f"cuda:{device_index}"), prompt_inputs["attention_mask"].to(f"cuda:{device_index}")
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            raise NotImplementedError
        else:
            # Regular generation path
            with torch.inference_mode():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        rewards_per_func = torch.zeros(len(completions_text), len(self.reward_funcs), device=device)
        rewards_per_token_func = torch.zeros(completion_ids.shape[0], completion_ids.shape[1], len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            for j in range(completion_ids.shape[1]):
                local_completion_ids = completion_ids.clone().tolist()
                # remove j-th token
                for k in range(len(local_completion_ids)):
                    local_completion_ids[k].pop(j)
                local_completion_ids = torch.Tensor(local_completion_ids).int()
                local_completions = self.processing_class.batch_decode(local_completion_ids, skip_special_tokens=True)
                local_output_reward_func = reward_func(prompts=prompts, completions=local_completions, **reward_kwargs)
                rewards_per_token_func[:, j, i] = torch.tensor(local_output_reward_func, dtype=torch.float32, device=device)
        
        rel_rewards_per_token_func = rewards_per_func.unsqueeze(1) - rewards_per_token_func

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        rel_rewards_per_token_func = gather(rel_rewards_per_token_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        rel_rewards_per_token = (rel_rewards_per_token_func * self.reward_weights.to(device).unsqueeze(0).unsqueeze(0)).sum(dim=2)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards_per_token = rel_rewards_per_token.mean(dim=1).view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards_per_token = rel_rewards_per_token.mean(dim=1).view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        mean_grouped_rewards_per_token = mean_grouped_rewards_per_token.repeat_interleave(self.num_generations, dim=0).unsqueeze(1).repeat(1, completion_ids.shape[1])
        std_grouped_rewards_per_token = std_grouped_rewards_per_token.repeat_interleave(self.num_generations, dim=0).unsqueeze(1).repeat(1, completion_ids.shape[1])
        advantages_per_token = (rel_rewards_per_token - mean_grouped_rewards_per_token) / (std_grouped_rewards_per_token + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        advantages_per_token = advantages_per_token[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "advantages_per_token": advantages_per_token,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        advantages_per_token = inputs["advantages_per_token"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) * advantages_per_token
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss

class FixedGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = prompts
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            raise NotImplementedError
        else:
            # Regular generation path
            with torch.inference_mode():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise NotImplementedError
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }