from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

mlp = model.transformer.h[0].mlp
col = 0
backup = mlp.c_proj.weight[:, col].clone()
mlp.c_proj.weight[:, col] *= 1.3
print(gen("The meaning of life is"))
mlp.c_proj.weight[:, col] = backup
