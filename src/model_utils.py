import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


from .inference_utils import generate

def init_train_model(config):
    model_name = config["model_name"]
    quant_config = None
    if "quant_config" in config:
        quant_args = {}
        if "load_in_4bit" in config["quant_config"]:
            quant_args["load_in_4bit"] = config["quant_config"]["load_in_4bit"]
        if "bnb_4bit_compute_dtype" in config["quant_config"]:
            value = None
            if config["quant_config"]["bnb_4bit_compute_dtype"] == "float16":
                value = torch.float16
            else:
                raise TypeError
            quant_args["bnb_4bit_compute_dtype"] = value
        if "bnb_4bit_quant_type" in config["quant_config"]:
            quant_args["bnb_4bit_quant_type"] = config["quant_config"]["bnb_4bit_quant_type"]
        if "bnb_4bit_use_double_quant" in config["quant_config"]:
            quant_args["bnb_4bit_use_double_quant"] = config["quant_config"]["bnb_4bit_use_double_quant"]
        quant_config = BitsAndBytesConfig(**quant_args)

    device_map = None
    if config["device_map"] == "current":
        device_index = Accelerator().process_index
        device_map = {"": device_index}
    else:
        device_map = config["device_map"]

    torch_dtype = None
    if config["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    else:
        raise TypeError

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quant_config,
                                                 device_map=device_map,
                                                torch_dtype=torch_dtype)
    return model

def init_inference_model(config):
    device_map = None
    if config["device_map"] == "current":
        device_index = Accelerator().process_index
        device_map = {"": device_index}
    else:
        device_map = config["device_map"]

    torch_dtype = None
    if config["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif config["torch_dtype"] == "auto":
        torch_dtype = "auto"
    else:
        raise TypeError
    
    #model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map=device_map, torch_dtype=torch_dtype, attn_implementation="flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map=device_map, torch_dtype=torch_dtype)
    #model.generation_config.cache_implementation = "static"
    model.generation_config.max_length = 8192
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    return model

