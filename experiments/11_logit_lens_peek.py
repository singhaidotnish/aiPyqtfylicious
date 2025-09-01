from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

layer_outs = []
def hook(_, __, out): layer_outs.append(out[:, -1, :].clone())
hk = model.transformer.h[0].register_forward_hook(hook)

_ = gen("The capital of France is")
hk.remove()

resid = layer_outs[0]  # last token’s residual after layer 0
W_u = model.lm_head.weight
logits = resid @ W_u.T
topk = torch.topk(logits, 10).indices[0]
print([tok.decode([i.item()]) for i in topk])
