import argparse

import torch

from .runtime import add_inference_arguments, inference_model
from .tokenizer import Tokenizer


@torch.inference_mode()
def generate(model, executable, prompt, max_new_tokens, eos_id=None, temperature=0.0):
    if max_new_tokens <= 0 or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature nonnegative")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt + max_new_tokens exceeds max_seq_len")
    cache = model.new_cache(prompt.shape[0], prompt.shape[1] + max_new_tokens)
    current = prompt
    outputs = []
    for _ in range(max_new_tokens):
        logits = executable(current, cache=cache, last_only=True)[:, -1, :].float()
        if temperature == 0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
        outputs.append(token)
        current = token
        if eos_id is not None and prompt.shape[0] == 1 and token.item() == eos_id:
            break
    return torch.cat((prompt, *outputs), dim=1)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a contiguous KV cache")
    add_inference_arguments(parser)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tokenizer", help="required with random-weight --config")
    args = parser.parse_args()
    model, executable, device, checkpoint = inference_model(args)
    if checkpoint:
        tokenizer = Tokenizer(checkpoint["tokenizer"])
    elif args.tokenizer:
        tokenizer = Tokenizer.load(args.tokenizer)
    else:
        parser.error("provide --checkpoint or --tokenizer")
    prompt = torch.tensor([tokenizer.encode(args.prompt, bos=True)], device=device)
    result = generate(model, executable, prompt, args.max_new_tokens, tokenizer.eos_id, args.temperature)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
