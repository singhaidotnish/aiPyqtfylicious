import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# load once
model_name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()

def gen(text: str, cfg_dict: dict | None = None, **overrides):
    """Generate text with safe, typed params."""
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    max_new_tokens = overrides.get("max_new_tokens", cfg_dict.get("max_new_tokens", 40))
    temperature    = overrides.get("temperature",    cfg_dict.get("temperature", 0.7))
    top_k          = overrides.get("top_k",          cfg_dict.get("top_k", 0))
    top_p          = overrides.get("top_p",          cfg_dict.get("top_p", 1.0))
    do_sample      = overrides.get("do_sample",      cfg_dict.get("do_sample", True))

    # hard-cast
    max_new_tokens = int(max_new_tokens)
    temperature    = float(temperature)
    top_k          = int(top_k)
    top_p          = float(top_p)
    do_sample      = bool(do_sample)

    ids = tok(text, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def compare(prompt: str, hook_ctx_fn=None, cfg_dict: dict | None = None):
    """Run base vs tweaked generation using a temporary hook."""
    base = gen(prompt, cfg_dict=cfg_dict)
    if hook_ctx_fn is None:
        return base, base
    h = hook_ctx_fn()
    try:
        tweaked = gen(prompt, cfg_dict=cfg_dict)
    finally:
        h.remove()
    return base, tweaked
