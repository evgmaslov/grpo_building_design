import json
from datasets import Dataset
from transformers import AutoTokenizer

def get_dataset(max_n_rooms):
    room_sets = get_room_sets(max_n_rooms)
    samples = []
    for room_set in room_sets:
        local_cons = get_connections(room_set)
        for local_con in local_cons:
            sample = {
                "rooms":room_set,
                "connections":[list(c) for c in local_con]
            }
            samples.append(sample)
    dataset_dict = {"prompt":[json.dumps(s) for s in samples]}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset
    
def get_room_sets(max_n_rooms):
    assert max_n_rooms > 3, "There cannot be less than 4 rooms."
    base_sets = [["hallway"]]

    #Add bathroom and toilet
    new_sets = []
    subsets = [["combined bathroom"], ["bathroom", "toilet"]]
    for subset in subsets:
        new_set = base_sets[0].copy()
        for room in subset:
            new_set.append(room)
        new_sets.append(new_set)
    base_sets = new_sets
    
    # Add kitchen and living room
    subsets = [["living room", "kitchen"], ["kitchen-living room"]]
    new_sets = []
    for base_set in base_sets:
        for subset in subsets:
            new_set = base_set.copy()
            for room in subset:
                new_set.append(room)
            new_sets.append(new_set)
    base_sets = new_sets
    base_sets_sets = [set(room_set) for room_set in base_sets]

    i = 0
    while i < len(base_sets):
        cur_set = base_sets[i]
        cur_types = [r.split("_")[0] for r in cur_set]
        new_sets = []
        if "corridor" not in cur_types:
            new_set = cur_set.copy()
            new_set.append("corridor")
            if len(new_set) < max_n_rooms:
                new_sets.append(new_set)

        if "wardrobe" not in cur_types: 
            new_set = cur_set.copy()
            new_set.append("wardrobe")
            if len(new_set) < max_n_rooms:
                new_sets.append(new_set)
        elif "bedroom" in cur_types:
            if len(new_set) < max_n_rooms:
                new_set = cur_set.copy()
                for ind in range(len(new_set)):
                    if new_set[ind] == "wardrobe":
                        new_set[ind] = "wardrobe_1"
                        break
                new_set.append("wardrobe_2")
                new_sets.append(new_set)
        
        n_cb = len([r for r in cur_set if r == "combined bathroom"])
        n_bathrooms = len([r for r in cur_set if r == "bathroom"])
        n_toilets = len([r for r in cur_set if r == "toilet"])
        counter = {
            "combined bathroom": n_cb,
            "bathroom": n_bathrooms,
            "toilet": n_toilets
        }
        if n_cb == 1 and n_bathrooms == 0 and n_toilets == 0 or n_cb == 0 and n_bathrooms == 1 and n_toilets == 1:
            subsets = [["combined bathroom"], ["toilet"], ["combined bathroom", "toilet"]]
            for subset in subsets:
                if len(new_set) < max_n_rooms - 1:
                    new_set = cur_set.copy()
                    for r in subset:
                        for ind in range(len(new_set)):
                            if new_set[ind] != r:
                                continue
                            if counter[new_set[ind]] > 0:
                                new_set[ind] = f"{new_set[ind]}_1"
                        if counter[r] > 0:
                            new_set.append(f"{r}_{counter[r] + 1}")
                        else:
                            new_set.append(r)
                    new_sets.append(new_set)
        
        if "bedroom" not in cur_types:
            max_n_bedrooms = 3
            for n in range(max_n_bedrooms):
                if len(new_set) < max_n_rooms - n:
                    new_set = cur_set.copy()
                    if n == 0:
                        new_set.append("bedroom")
                    else:
                        for ind in range(n + 1):
                            new_set.append(f"bedroom_{ind + 1}")
                    new_sets.append(new_set)
        
        new_sets_cleaned = []
        new_sets_cleaned_sets = []
        for new_set in new_sets:
            new_set_set = set(new_set)
            if new_set_set not in base_sets_sets:
                new_sets_cleaned.append(new_set)
                new_sets_cleaned_sets.append(new_set_set)

        base_sets = base_sets + new_sets_cleaned
        base_sets_sets = base_sets_sets + new_sets_cleaned_sets
        i += 1
    
    return base_sets

def get_connections(room_set):
    connections = [[]]

    base_rooms = ["corridor"] if "corridor" in room_set else ["hallway"]
    living_room = "living room" if "living room" in room_set else "kitchen-living room"
    base_rooms.append(living_room)
    
    con_vars = {k: [] for k in room_set if k not in base_rooms}
    bathrooms = [r for r in list(room_set) if r.startswith("combined bathroom") or r.startswith("toilet") or r.startswith("bathroom")]
    wardrobes = [r for r in list(room_set) if r.startswith("wardrobe")]
    non_bathrooms = [r for r in list(room_set) if r not in bathrooms and r not in wardrobes and r not in base_rooms]

    if "corridor" in room_set:
        con_vars["corridor"] = [["hallway", "corridor"], [living_room, "corridor"]]

    for room in non_bathrooms:
        for base_room in base_rooms:
            con_vars[room].append((room, base_room))
    
    for room in bathrooms:
        room_split = room.split("_")
        bathrooms_base_rooms = base_rooms.copy() + ["hallway"]
        bedrooms = [r for r in non_bathrooms if r.startswith("bedroom")]
        if len(room_split) > 1:
            if room_split[0] == "combined bathroom" and int(room_split[1]) == 2 and len(bedrooms) > 0:
                local_bedroom = "bedroom" if "bedroom" in bedrooms else "bedroom_1"
                bathrooms_base_rooms.append(local_bedroom)
        else:
            if len(bathrooms) > 1 and len(bedrooms) > 0 and room_split[0] == "combined bathroom":
                local_bedroom = "bedroom" if "bedroom" in bedrooms else "bedroom_1"
                bathrooms_base_rooms.append(local_bedroom)

        for base_room in bathrooms_base_rooms:
            con_vars[room].append((room, base_room))
    
    for room in wardrobes:
        room_split = room.split("_")
        wardrobes_base_rooms = ["hallway"]
        if len(room_split) > 1:
            bedrooms = [r for r in non_bathrooms if r.startswith("bedroom")]
            if int(room_split[1]) == 2 and len(bedrooms) > 0:
                local_bedroom = "bedroom" if "bedroom" in bedrooms else "bedroom_1"
                wardrobes_base_rooms = [local_bedroom]
        for base_room in wardrobes_base_rooms:
            con_vars[room].append((room, base_room))

    for room in con_vars.keys():
        new_connnections = []
        for cons in connections:
            for con_var in con_vars[room]:
                new_cons = cons.copy()
                con = set(con_var)
                if con not in new_cons:
                    new_cons.append(con)
                new_connnections.append(new_cons)
        connections = new_connnections
    return connections

def get_prompt(sample, tokenizer, base_system_prompt, few_shot_template, few_shots, user_prompt_template):
    system_prompt = base_system_prompt
    if len(few_shots) > 0:
        templated_few_shots = []
        for i, shot in enumerate(few_shots):
            args = {k:shot[k] for k in shot.keys()}
            shot_formatted = few_shot_template.format(**args)
            templated_few_shots.append(shot_formatted)
        templated_few_shots = "\n".join(templated_few_shots)
        system_prompt = system_prompt + templated_few_shots
    user_prompt = user_prompt_template.format(sample=sample)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def init_dataset(config):
    max_n_rooms = config["max_n_rooms"]
    test_size = config["test_size"]
    tokenizer_name = config["tokenizer_name"]
    base_system_prompt = config["base_system_prompt"]
    few_shot_template = config["few_shot_template"]
    few_shots = config["few_shots"]
    user_prompt_template = config["user_prompt_template"]

    dataset = get_dataset(max_n_rooms)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = dataset.map(lambda row: {"prompt":get_prompt(row["prompt"], tokenizer, base_system_prompt, few_shot_template, few_shots, user_prompt_template)})
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    return dataset
