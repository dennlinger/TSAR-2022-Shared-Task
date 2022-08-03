import torch
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


def load_6b_model():
    device = 1 if torch.cuda.is_available() else "cpu"

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                            torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    prompt = "Give a synonym for the following word: compulsory\nSynonym:"

    print(pipe(prompt, do_sample=True, temperature=0.6, max_length=20))


if __name__ == '__main__':
    checkpoint = "EleutherAI/gpt-neox-20b"
    config = AutoConfig.from_pretrained(checkpoint)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = load_checkpoint_and_dispatch(
        model, "/home/daumiller/gpt-neox-20b",
        device_map="auto", no_split_module_classes = ["GPTNeoXLayer"],
        offload_folder="/home/daumiller/gpt-offload/"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer("Find a synonym for the following word: compulsory\nSynonym:", return_tensors="pt")
    inputs = inputs.to(0)
    output = model.generate(inputs["input_ids"])
    print(tokenizer.decode(output[0].tolist()))

