# Very rough: scale the output projection slice associated with a “copy head”

from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

proj = model.transformer.h[0].attn.c_proj.weight
backup = proj.clone()
proj *= 1.05
print(gen("Repeat after me: namaste"))
model.transformer.h[0].attn.c_proj.weight = backup
