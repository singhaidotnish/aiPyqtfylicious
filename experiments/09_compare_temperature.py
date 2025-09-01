from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

ids = tok("Write a tagline for a bike tour:", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=25, temperature=0.2, top_k=0)
print("T=0.2:", tok.decode(out[0], skip_special_tokens=True))
out = model.generate(**ids, max_new_tokens=25, temperature=1.2, top_k=40)
print("T=1.2:", tok.decode(out[0], skip_special_tokens=True))
