import json
from datasets import Dataset

def generate_dataset():
    room_sets = get_room_sets()
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
    
def get_room_sets():
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
            new_sets.append(new_set)

        if "wardrobe" not in cur_types: 
            new_set = cur_set.copy()
            new_set.append("wardrobe")
            new_sets.append(new_set)
        elif "bedroom" in cur_types:
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

FEW_SHOTS = [
    {
        "requirements": json.dumps({
            "rooms":["hallway", "combined bathroom", "kitchen-living room"],
            "connections":[["hallway", "combined bathroom"], ["hallway", "kitchen-living room"]]
        }),
        "flat":json.dumps({
            "rooms":[
                {"name":"hallway",
                 "walls":[
                    {"start_point": [0, 0], "end_point": [0, 1610], "doors_to":["combined bathroom"]},
                    {"start_point": [0, 1610], "end_point": [2000, 1610], "doors_to":[]},
                    {"start_point": [2000, 1610], "end_point": [2000, 0], "doors_to":[]},
                    {"start_point": [2000, 0], "end_point": [0, 0], "doors_to":["kitchen-living room"]},
                 ]},
                {"name":"combined bathroom",
                 "walls":[
                    {"start_point": [-1850, 0], "end_point": [-1850, 1610], "doors_to":[]},
                    {"start_point": [-1850, 1610], "end_point": [0, 1610], "doors_to":[]},
                    {"start_point": [0, 1610], "end_point": [0, 0], "doors_to":["hallway"]},
                    {"start_point": [0, 0], "end_point": [-1850, 0], "doors_to":[]},
                 ]},
                {"name":"kitchen-living room",
                 "walls":[
                    {"start_point": [-1850, 0], "end_point": [2000, 0], "doors_to":["hallway"]},
                    {"start_point": [2000, 0], "end_point": [2000, -5390], "doors_to":[]},
                    {"start_point": [2000, -5390], "end_point": [-1850, -5390], "doors_to":[]},
                    {"start_point": [-1850, -5390], "end_point": [-1850, 0], "doors_to":[]},
                 ]}
            ]
        })
    },
    {
        "requirements": json.dumps({
            "rooms":["hallway", "wardrobe_1", "toilet", "kitchen", "living room", "bedroom", "wardrobe_2", "combined bathroom"],
            "connections":[["hallway", "kitchen"], ["hallway", "wardrobe_1"], ["hallway", "toilet"], ["kitchen", "living room"], ["living room", "bedroom"], ["bedroom", "wardrobe_2"], ["wardrobe_2", "combined bathroom"]]
        }),
        "flat": json.dumps({
            "rooms":[
                {"name":"hallway",
                 "walls":[
                    {"start_point": [0, 0], "end_point": [0, 2870], "doors_to":[]},
                    {"start_point": [0, 2870], "end_point": [1940, 2870], "doors_to":["kitchen"]},
                    {"start_point": [1940, 2870], "end_point": [1940, 0], "doors_to":["wardrobe_1", "toilet"]},
                    {"start_point": [1940, 0], "end_point": [0, 0], "doors_to":[]},
                 ]},
                 {"name":"wardrobe_1",
                 "walls":[
                    {"start_point": [1940, 0], "end_point": [1940, 1520], "doors_to":["hallway"]},
                    {"start_point": [1940, 1520], "end_point": [5650, 1520], "doors_to":[]},
                    {"start_point": [5650, 1520], "end_point": [5650, 0], "doors_to":[]},
                    {"start_point": [5650, 0], "end_point": [1940, 0], "doors_to":[]},
                 ]},
                 {"name":"toilet",
                 "walls":[
                    {"start_point": [1940, 1520], "end_point": [1940, 2870], "doors_to":["hallway"]},
                    {"start_point": [1940, 2870], "end_point": [5650, 2870], "doors_to":[]},
                    {"start_point": [5650, 2870], "end_point": [5650, 1520], "doors_to":[]},
                    {"start_point": [5650, 1520], "end_point": [1940, 1520], "doors_to":[]},
                 ]},
                 {"name":"kitchen",
                 "walls":[
                    {"start_point": [0, 2870], "end_point": [0, 7300], "doors_to":[]},
                    {"start_point": [0, 7300], "end_point": [1940, 7300], "doors_to":[]},
                    {"start_point": [1940, 7300], "end_point": [1940, 2870], "doors_to":["living room"]},
                    {"start_point": [1940, 2870], "end_point": [0, 2870], "doors_to":["hallway"]},
                 ]},
                 {"name":"living room",
                 "walls":[
                    {"start_point": [1940, 2870], "end_point": [1940, 7300], "doors_to":["kitchen"]},
                    {"start_point": [1940, 7300], "end_point": [5650, 7300], "doors_to":[]},
                    {"start_point": [5650, 7300], "end_point": [5650, 2870], "doors_to":["bedroom"]},
                    {"start_point": [5650, 2870], "end_point": [1940, 2870], "doors_to":[]},
                 ]},
                 {"name":"bedroom",
                 "walls":[
                    {"start_point": [5650, 2870], "end_point": [5650, 7300], "doors_to":["living room"]},
                    {"start_point": [5650, 7300], "end_point": [8850, 7300], "doors_to":[]},
                    {"start_point": [8850, 7300], "end_point": [8850, 2870], "doors_to":[]},
                    {"start_point": [8850, 2870], "end_point": [5650, 2870], "doors_to":["wardrobe_2"]},
                 ]},
                 {"name":"wardrobe_2",
                 "walls":[
                    {"start_point": [7360, 0], "end_point": [7360, 2870], "doors_to":["combined bathroom"]},
                    {"start_point": [7360, 2870], "end_point": [8850, 2870], "doors_to":["bedroom"]},
                    {"start_point": [8850, 2870], "end_point": [8850, 0], "doors_to":[]},
                    {"start_point": [8850, 0], "end_point": [7360, 0], "doors_to":[]},
                 ]},
                 {"name":"combined bathroom",
                 "walls":[
                    {"start_point": [5650, 0], "end_point": [5650, 2870], "doors_to":[]},
                    {"start_point": [5650, 2870], "end_point": [7360, 2870], "doors_to":[]},
                    {"start_point": [7360, 2870], "end_point": [7360, 0], "doors_to":["wardrobe_2"]},
                    {"start_point": [7360, 0], "end_point": [5650, 0], "doors_to":[]},
                 ]},
            ]
        })
    },
]

def get_prompt(sample, tokenizer):
    base_prompt = """You are an engineer architect who designs an apartment. You create a plan for the apartment, determine the location of the walls and doors. You arrange the rooms and determine the connections between them. 
    You receive a set of requirements for the apartment, which include a set of rooms and connections between them. Create an apartment plan in json format, in which you indicate the walls for each room. Also indicate doors in the walls if you need to connect adjacent rooms.
    Here are some examples of what you need to do:
    """
    base_prompt = """You receive a set of requirements for the apartment, which include a set of rooms and connections between them. Create an apartment plan in json format, in which you indicate the walls for each room. Also indicate doors in the walls if you need to connect adjacent rooms.
    The apartment plan should be a json dictionary with a single key “rooms”, which will contain a list of rooms. Each room is also a dictionary with two keys: "name", which contains the name of the room, and "walls", which contains a list of the walls that form the room. Each wall is a dictionary with three keys: "start_point", which contains a list of two coordinates (x and y) of the starting point of the wall, "end_point", containing the coordinates of the end point of the wall, "doors_to", containing a list of the names of the rooms to which the doors located in this wall lead. The walls are arranged in an order that forms a closed loop. This means that the starting point of the wall must coincide with the ending point of the previous wall.
    Enclose the answer in <answer> and </answer> tokens. Before answering, write down your reasoning, surrounding it with <think> and </think> tokens.
    Here are an example of what you need to do:
    """

    #Actual prompt
    base_prompt = """You receive a set of requirements for the apartment, which include a set of rooms and connections between them. Create an apartment plan in json format, in which you indicate the walls for each room. Also indicate doors in the walls if you need to connect adjacent rooms.
    Enclose the answer in <answer> and </answer> tokens. Before answering, write down your reasoning, surrounding it with <think> and </think> tokens.
    Here are an example of what you need to do:
    """

    template = """Example {n_example}
    Requirements: {requirements}
    Flat: {flat}"""
    few_shots = []
    for i, shot in enumerate(FEW_SHOTS):
        args = {k:shot[k] for k in shot.keys()}
        args["n_example"] = str(i + 1)
        shot_formatted = template.format(**args)
        few_shots.append(shot_formatted)
    few_shots = "\n".join(few_shots)
    system_prompt = base_prompt + few_shots
    user_prompt = sample

    system_prompt = base_prompt
    user_prompt = sample

    #Actual few shots
    template = """Requirements: {requirements}
    Flat: <think>Chain of thoughts</think><answer>{flat}</answer>"""
    few_shots = []
    for i, shot in enumerate(FEW_SHOTS[1:]):
        args = {k:shot[k] for k in shot.keys()}
        shot_formatted = template.format(**args)
        few_shots.append(shot_formatted)
    few_shots = "\n".join(few_shots)
    system_prompt = base_prompt + few_shots
    user_prompt = f"""Requirements: {sample}
    Flat: """

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
