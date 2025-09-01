from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

proj = model.transformer.h[0].attn.c_proj.weight
heads, dim = model.config.n_head, model.config.n_embd
per = dim // heads
h = 0; s, e = h*per, (h+1)*per
backup = proj[:, s:e].clone()
proj[:, s:e] *= 1.2
print(gen("Explain photosynthesis in one line."))
proj[:, s:e] = backup
