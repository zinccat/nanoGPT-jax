import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from einops import rearrange
import numpy as np
from functools import partial

batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3
eval_interval = 500
eval_iters = 200
e_embed = 32

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

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, data.shape[0] - block_size, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

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

class Head(nn.Module):
    head_size: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        k = nn.Dense(self.head_size, name='key', use_bias=False)(x)
        q = nn.Dense(self.head_size, name='query', use_bias=False)(x)
        v = nn.Dense(self.head_size, name='value', use_bias=False)(x)

        wei = q @ rearrange(k, 'b t h -> b h t') * (self.head_size ** -0.5)
        tri = jnp.triu(jnp.full((T, T), -jnp.inf), k=1) # might need to change this
        wei = wei + tri
        wei = jax.nn.softmax(wei, axis=-1)
        out = wei @ v
        return out

class BigramLanguageModel(nn.Module):
    vocab_size: int
    n_embed: int = 32
    
    @nn.compact
    def __call__(self, idx: jnp.ndarray, targets: jnp.ndarray = None):
        B, T = idx.shape

        tok_emb = nn.Embed(num_embeddings=self.vocab_size, name='token_embedding_table', features=self.n_embed)(idx) # (B, T, C)
        pos_emb = nn.Embed(num_embeddings=block_size, name='position_embedding_table', features=self.n_embed)(jnp.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = Head(head_size=e_embed)(x)
        logits = nn.Dense(self.vocab_size, name='lm_head')(tok_emb) # (B, T, vocab_size)
        if targets is None:
            return logits, 0
        logits_reshaped = rearrange(logits, 'b t c -> (b t) c')
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_reshaped, targets.flatten())
        return logits, jnp.mean(loss)

    def generate(self, params, subkey, idx, max_new_tokens: int = 100):
        # pad idx to max_new_tokens
        idx = jnp.pad(idx, ((0, 0), (0, max_new_tokens)), mode='constant', constant_values=0)

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

optimizer = optax.adamw(learning_rate)
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
# for _ in range(1000):
#     params, opt_state, loss = train_step(params, opt_state, xb, yb)
#     jax.block_until_ready(loss)
# end_time = timer()
# print("Time taken:", end_time - start_time)

for step in range(max_iters):
    xb, yb = get_batch("train")
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    
    if step % eval_interval == 0:
        loss = estimate_loss()
        print(f"Step {step}, Loss: {loss}", "Elapsed time:", timer() - start_time)
        idx = jnp.zeros((1, 1), dtype=jnp.int32)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
subkey, key = jax.random.split(subkey)
print(decode(np.array(m.generate(params, key, idx, max_new_tokens=500)[0])))