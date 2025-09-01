from experiments.quick_setup_run_once import model, gen
import torch, difflib

def ablate_head(layer: int, head_idx: int):
    heads, dim = model.config.n_head, model.config.n_embd
    per = dim // heads
    s, e = head_idx * per, (head_idx + 1) * per
    def hook(_m, _i, out):
        out2 = out.clone()
        out2[..., s:e] = 0.0
        return out2
    return model.transformer.h[layer].attn.c_proj.register_forward_hook(hook)

def delta_score(a: str, b: str) -> float:
    # 1 - similarity ratio (0 = same, 1 = very different)
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()

def main(cfg_dict: dict):
    prompt = cfg_dict["prompts"]["capital"]
    base = gen(prompt, cfg_dict=cfg_dict)

    layer = 0
    heads = model.config.n_head
    results = []
    for h in range(heads):
        hk = ablate_head(layer, h)
        try:
            out = gen(prompt, cfg_dict=cfg_dict)
        finally:
            hk.remove()
        score = delta_score(base, out)
        results.append((h, score, out))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"[Layer {layer}] most influential heads by Δ-score:")
    for h, s, _ in results:
        print(f"  head {h:02d}: Δ={s:.3f}")

    # Show the biggest change
    h, s, out = results[0]
    print("\n--- TOP CHANGE ---")
    print(f"Head {h} Δ={s:.3f}")
    print("BASE   :", base)
    print("TWEAKED:", out)
