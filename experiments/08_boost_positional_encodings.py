from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

wpe = model.transformer.wpe.weight
backup = wpe.clone()
wpe[:] = wpe * 1.5
print(gen("List three reasons bicycles are great:"))
wpe[:] = backup
