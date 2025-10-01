Tokenization

The text is broken into tokens (“How”, “to”, “prepare”, “thread”, …).

Each token becomes an embedding vector.


Context processing (transformer layers)

Self-attention looks at relationships: e.g. “thread” relates to “stitching” more strongly than to “How”.

Positional encodings make sure order matters (from 08_boost_positional_encodings.py


Hidden representations

Attention heads each focus on different aspects (maybe one head attends to “thread ↔ stitching”).

MLP neurons push toward certain semantic directions (from 04_nudge_one_mlp_neuron.py


Next-token prediction

The model computes probabilities for the next token (“You”, “First”, “Cut”, …).

Logit bias, temperature, and sampling decide which token is chosen (shown in 09_compare_temperature.py


Iteration

This repeats until the model generates a full answer.
