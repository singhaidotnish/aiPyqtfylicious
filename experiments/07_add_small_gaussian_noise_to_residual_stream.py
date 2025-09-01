from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

def noise_hook(_, __, out):
    return out + 0.02*torch.randn_like(out)

hk = model.transformer.h[0].register_forward_hook(noise_hook)
print(gen("Explain gravity in simple terms:"))
hk.remove()
