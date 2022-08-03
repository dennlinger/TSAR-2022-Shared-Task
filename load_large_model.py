import torch
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline


if __name__ == '__main__':

    device = 1 if torch.cuda.is_available() else "cpu"

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    prompt = "Give a synonym for the following word: compulsory\nSynonym:"

    print(pipe(prompt, do_sample=True, temperature=0.6, max_length=20))
