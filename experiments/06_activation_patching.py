from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))


clean = tok("Paris is the capital of", return_tensors="pt")
corrupt = tok("Bananas are a type of", return_tensors="pt")

acts = {}
def cap(name):
    def hook(m, i, o): acts[name] = o.clone()
    return hook

h = model.transformer.h[0].mlp.register_forward_hook(cap("mlp0"))

_ = model(**clean)     # record clean activation
h.remove()

def patch_hook(m, i, o): return acts["mlp0"]
h2 = model.transformer.h[0].mlp.register_forward_hook(patch_hook)
print(gen("Bananas are a type of"))   # uses clean mlp activation
h2.remove()
