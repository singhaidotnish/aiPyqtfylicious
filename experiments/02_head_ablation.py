from experiments.quick_setup_run_once import model, gen, compare
import torch  # <-- needed for torch.arange

def ablate_head(layer: int = 0, head_idx: int = 0):
    """Register a forward hook that zeros one attention head's contribution."""
    heads, dim = model.config.n_head, model.config.n_embd
    per = dim // heads
    s, e = head_idx * per, (head_idx + 1) * per

    def hook(_m, _inp, out):
        out2 = out.clone()
        out2[..., s:e] = 0.0
        return out2

    return model.transformer.h[layer].attn.c_proj.register_forward_hook(hook)


def noop_hook():
    def _noop(m, i, o): return o
    return model.transformer.h[0].attn.c_proj.register_forward_hook(_noop)


def main(cfg_dict: dict):
    prompt = cfg_dict["prompts"]["capital"]
    # compare() returns (base, tweaked)
    base, tweaked = compare(
        prompt,
        # hook_ctx_fn=lambda: ablate_head(layer=0, head_idx=0),
        hook_ctx_fn=noop_hook,
        cfg_dict=cfg_dict,
    )
    print("BASE   :", base)
    print("TWEAKED:", tweaked)