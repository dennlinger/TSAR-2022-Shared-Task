import torch
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map


def run_6b_model():
    device = 1 if torch.cuda.is_available() else "cpu"

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                            torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    prompt = "Give a synonym for the following word: compulsory\nSynonym:"

    print(pipe(prompt, do_sample=True, temperature=0.6, max_length=20))


def run_20b_model():
    checkpoint = "EleutherAI/gpt-neox-20b"
    config = AutoConfig.from_pretrained(checkpoint)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = load_checkpoint_and_dispatch(
        model, "/home/daumiller/gpt-neox-20b",
        device_map="auto", no_split_module_classes=["GPTNeoXLayer"],
        offload_folder="/home/daumiller/gpt-offload/"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer("Find a synonym for the following word: compulsory\nSynonym:", return_tensors="pt")
    inputs = inputs.to(0)
    output = model.generate(inputs["input_ids"])
    print(tokenizer.decode(output[0].tolist()))


if __name__ == '__main__':

    checkpoint = "facebook/opt-30b"
    config = AutoConfig.from_pretrained(checkpoint)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    print(infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer", "Embedding"]))
    # Based on the auto map, but extended with explicit mappings for error-causing layers.
    custom_opt_map = {
        'model.decoder.embed_tokens': 0,
        'model.decoder.embed_positions': 0,
        'model.decoder.final_layer_norm': 0,
        'model.decoder.layers.0': 0,
        'model.decoder.layers.1': 0,
        'model.decoder.layers.2': 0,
        'model.decoder.layers.3': 0,
        'model.decoder.layers.4': 0,
        'model.decoder.layers.5': 0,
        'model.decoder.layers.6': 1,
        'model.decoder.layers.7': 1,
        'model.decoder.layers.8': 1,
        'model.decoder.layers.9': 1,
        'model.decoder.layers.10': 1,
        'model.decoder.layers.11': 1,
        'model.decoder.layers.12': 1,
        'model.decoder.layers.13': 1,
        'model.decoder.layers.14': 'cpu',
        'model.decoder.layers.15': 'cpu',
        'model.decoder.layers.16': 'cpu',
        'model.decoder.layers.17': 'cpu',
        'model.decoder.layers.18': 'cpu',
        'model.decoder.layers.19': 'cpu',
        'model.decoder.layers.20': 'cpu',
        'model.decoder.layers.21': 'cpu',
        'model.decoder.layers.22': 'cpu',
        'model.decoder.layers.23': 'cpu',
        'model.decoder.layers.24': 'cpu',
        'model.decoder.layers.25': 'cpu',
        'model.decoder.layers.26': 'cpu',
        'model.decoder.layers.27': 'cpu',
        'model.decoder.layers.28': 'cpu',
        'model.decoder.layers.29': 'cpu',
        'model.decoder.layers.30': 'cpu',
        'model.decoder.layers.31': 'cpu',
        'model.decoder.layers.32': 'disk',
        'model.decoder.layers.33': 'disk',
        'model.decoder.layers.34': 'disk',
        'model.decoder.layers.35': 'disk',
        'model.decoder.layers.36': 'disk',
        'model.decoder.layers.37': 'disk',
        'model.decoder.layers.38': 'disk',
        'model.decoder.layers.39': 'disk',
        'model.decoder.layers.40': 'disk',
        'model.decoder.layers.41': 'disk',
        'model.decoder.layers.42': 'disk',
        'model.decoder.layers.43': 'disk',
        'model.decoder.layers.44': 'disk',
        'model.decoder.layers.45': 'disk',
        'model.decoder.layers.46': 'disk',
        'model.decoder.layers.47': 'disk',
        'lm_head': 'disk'
    }

    model = load_checkpoint_and_dispatch(
        model, "/home/daumiller/opt-30b",
        device_map=custom_opt_map, no_split_module_classes=["OPTDecoderLayer"],
        offload_folder="/home/daumiller/gpt-offload/"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer("Find a synonym for the following word: compulsory\nSynonym:", return_tensors="pt")
    inputs = inputs.to(0)
    output = model.generate(inputs["input_ids"])
    print(tokenizer.decode(output[0].tolist()))
