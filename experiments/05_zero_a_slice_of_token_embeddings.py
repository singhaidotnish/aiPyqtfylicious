from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

emb = model.transformer.wte.weight
sl = slice(0, emb.shape[1]//8)
backup = emb[:, sl].clone()
emb[:, sl] = 0
print(gen("Write a short poem about rain in Mumbai."))
emb[:, sl] = backup
