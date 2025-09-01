from quick_setup_run_once import model
import matplotlib.pyplot as plt

attn = model.transformer.h[0].attn
Wqkv = attn.c_attn.weight.cpu()           # [embed, 3*embed]
d = model.config.n_embd
Wq = Wqkv[:, :d]

plt.figure(figsize=(6,4))
plt.imshow(Wq[:128, :128], aspect='auto', cmap='coolwarm')
plt.title("Layer 0: Wq (partial heatmap)")
plt.colorbar()
plt.show()
