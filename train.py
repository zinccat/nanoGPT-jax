import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from einops import rearrange
import numpy as np
from functools import partial

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)
# print("vocab_size:", vocab_size)
# print(''.join(chars))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

data = np.array(encode(text), dtype=np.int32) # no default int64, use numpy for underlying data

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

rng = jax.random.PRNGKey(42)
key, subkey = jax.random.split(rng)

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, data.shape[0] - block_size, batch_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, idx: jnp.ndarray, targets: jnp.ndarray = None):
        logits = nn.Embed(num_embeddings=self.vocab_size, features=self.vocab_size)(idx) # (bs, block_size, 
        if targets is None:
            return logits, 0
        logits_reshaped = rearrange(logits, 'b t c -> (b t) c')
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_reshaped, targets.flatten())
        return logits, jnp.mean(loss)

    def generate(self, params, subkey, idx, max_new_tokens: int = 100):
        
        def step_fn(idx: jnp.ndarray, key: jnp.ndarray):
            logits, loss = self.apply(params, idx)
            logits = logits[:, -1, :] # we are not using the logits from the rest of the sequence
            # probs = jax.nn.softmax(logits)
            token = jax.random.categorical(key, logits)[:, None]
            new_idx = jnp.concatenate([idx, token], axis=-1)
            return new_idx
        for _ in range(max_new_tokens):
            subkey, key = jax.random.split(subkey)
            idx = step_fn(idx, key)

        # use scan to generate tokens
        # subkey, key = jax.random.split(subkey)
        # idx = jax.lax.scan(step_fn, idx, jnp.arange(max_new_tokens), length=max_new_tokens, reverse=False)
        # idx = jax.lax.scan(step_fn, key, idx, jnp.arange(max_new_tokens))
        return jnp.array(idx)

xb, yb = get_batch("train")

m = BigramLanguageModel(vocab_size=vocab_size)
params = m.init(subkey, xb, yb)
xb_, yb_ = get_batch("train")
out, loss = m.apply(params, xb_, yb_)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
# print(decode(np.array(m.generate(params, key, idx, max_new_tokens=500)[0])))

optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(params)
from timeit import default_timer as timer

@jax.jit
def train_step(params, opt_state, xb, yb):
    def loss_fn(params):
        logits, loss = m.apply(params, xb, yb)
        return loss
    
    loss, grad = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

start_time = timer()

for step in range(10000):
    xb, yb = get_batch("train")
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    
    if step % 100 == 0:  # Print loss every 100 steps
        jax.block_until_ready(loss)
        print(f"Step {step}, Loss: {loss:.4f}", "Elapsed time:", timer() - start_time)
        idx = jnp.zeros((1, 1), dtype=jnp.int32)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
subkey, key = jax.random.split(subkey)
print(decode(np.array(m.generate(params, key, idx, max_new_tokens=100)[0])))