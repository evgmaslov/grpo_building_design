import re
import json
from shapely.geometry import Polygon
import random
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import torch
import wandb

from .data_utils import FEW_SHOTS

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
        pattern_1 = "<think>.*?</think>"
        if len(re.findall(pattern_1, c)) == 1:
            reward += 1/8
        pattern_2 = "<answer>.*?</answer>"
        if len(re.findall(pattern_2, c)) == 1:
            reward += 1/8
        if "</think><answer>" in c:
            reward += 1/8
        pattern_3 = "<think>.*?</think><answer>.*?</answer>"
        if len(re.findall(pattern_3, c)) == 1:
            reward += 1/8
        rewards.append(reward)
    return rewards

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

def answer_format_reward(**kwargs):
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
            pattern = "(?<=\</think\>\<answer>).*?(?=\</answer\>)"
            answer = re.findall(pattern, c)[0]
            #answer = c
            try:
                answer_json = json.loads(answer)
            except:
                rewards.append(0)
                continue
            
            answer_format = json.loads(FEW_SHOTS[0]["flat"])
            result = check_format(answer_json, answer_format)
            rewards.append(result)
        except:
            rewards.append(0)
    return rewards

def walls_consistency_reward(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = answer_format_reward(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\</think\>\<answer>).*?(?=\</answer\>)"
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

def doors_consistency_reward(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = answer_format_reward(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\</think\>\<answer>).*?(?=\</answer\>)"
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

def geometry_consistency_reward(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = walls_consistency_reward(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            pattern = "(?<=\</think\>\<answer>).*?(?=\</answer\>)"
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

def prompt_consistency_reward(**kwargs):
    prompts = kwargs["prompts"]
    completions = kwargs["completions"]
    rewards = []
    base_rewards = doors_consistency_reward(**kwargs)
    for i in range(len(completions)):
        if base_rewards[i] < 1:
            rewards.append(0)
            continue
        try:
            c = completions[i]
            p = prompts[i]
            pattern = "(?<=\</think\>\<answer>).*?(?=\</answer\>)"
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

def get_inference_text(prompt, text, tokenizer):
  messages = [
      {"role": "system", "content": prompt},
      {"role": "user", "content": text}
  ]
  input_text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  return input_text

def generate(model, tokenizer, generation_config, inp):
  tokenized_prompt = tokenizer(inp, return_tensors='pt', padding=True, truncation=True)
  tokenized_prompt_ids = tokenized_prompt["input_ids"].cuda()
  tokenized_prompt_mask = tokenized_prompt["attention_mask"].cuda()
  with torch.inference_mode():
      output = model.generate(**{"input_ids":tokenized_prompt_ids, "attention_mask":tokenized_prompt_mask, "generation_config":generation_config}).detach().cpu()
  decoded = []
  for i in range(output.shape[0]):
    ans = tokenizer.decode(output[i][len(tokenized_prompt[0]):], skip_special_tokens=True)
    decoded.append(ans)
  return decoded

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