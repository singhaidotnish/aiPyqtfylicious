from quick_setup_run_once import model, gen
print(gen("Paris is the capital of"))

bias = model.lm_head.bias
backup = bias.clone()
comma_id = tok(",")["input_ids"][0]
bias[comma_id] += 1.5
print(gen("Describe the monsoon in Mumbai"))
model.lm_head.bias = backup
