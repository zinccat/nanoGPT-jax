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

batch_size = 32
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, data.shape[0] - block_size, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

eval_iters = 20
def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, l = m.apply(params, xb, yb)
            loss += l
        out[split] = (loss / eval_iters).item()
    return out

class BigramLanguageModel(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, idx: jnp.ndarray, targets: jnp.ndarray = None):
        logits = nn.Embed(num_embeddings=self.vocab_size, features=self.vocab_size)(idx) # (bs, block_size, vocab_size)
        if targets is None:
            return logits, 0
        logits_reshaped = rearrange(logits, 'b t c -> (b t) c')
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_reshaped, targets.flatten())
        return logits, jnp.mean(loss)

    def generate(self, params, subkey, idx, max_new_tokens: int = 100):
        # pad idx to max_new_tokens
        idx = jnp.pad(idx, ((0,0), (0, max_new_tokens)), mode='constant', constant_values=0)

        @jax.jit
        def step_fn(i: int, idx: jnp.ndarray, key: jnp.ndarray):
            logits, loss = self.apply(params, idx)
            logits = logits[:, i, :] # we are not using the logits from the rest of the sequence
            token = jax.random.categorical(key, logits)
            new_idx = idx.at[:, i+1].set(token)
            return new_idx
        for i in range(max_new_tokens):
            subkey, key = jax.random.split(subkey)
            idx = step_fn(i, idx, key)
        return jnp.array(idx)

xb, yb = get_batch("train")

m = BigramLanguageModel(vocab_size=vocab_size)
params = m.init(subkey, xb, yb)
xb_, yb_ = get_batch("train")
out, loss = m.apply(params, xb_, yb_)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(np.array(m.generate(params, key, idx, max_new_tokens=500)[0])))

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
for _ in range(1000):
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    jax.block_until_ready(loss)
end_time = timer()
print("Time taken:", end_time - start_time)

for step in range(10000):
    xb, yb = get_batch("train")
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    
    if step % 2000 == 0:  # Print loss every 100 steps
        # jax.block_until_ready(loss)
        loss = estimate_loss()
        print(f"Step {step}, Loss: {loss}", "Elapsed time:", timer() - start_time)
        idx = jnp.zeros((1, 1), dtype=jnp.int32)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
subkey, key = jax.random.split(subkey)
print(decode(np.array(m.generate(params, key, idx, max_new_tokens=500)[0])))