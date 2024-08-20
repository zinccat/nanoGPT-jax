import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from einops import rearrange
import numpy as np
from functools import partial
from timeit import default_timer as timer

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10 #200
n_embed = 384
n_ff = 4 * n_embed
n_head = 6
n_layer = 6
dropout_rate = 0.2
max_seq_len = 1000
dtype = jnp.bfloat16

rng = jax.random.PRNGKey(1337)
dropout_rng = jax.random.PRNGKey(42)
np.random.seed(1337)

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

key, subkey = jax.random.split(rng)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, data.shape[0] - block_size, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class Head(nn.Module):
    head_size: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T, C = x.shape
        k = nn.Dense(self.head_size, name='key', use_bias=False, dtype=self.dtype, kernel_init=initializer)(x)
        q = nn.Dense(self.head_size, name='query', use_bias=False, dtype=self.dtype, kernel_init=initializer)(x)
        v = nn.Dense(self.head_size, name='value', use_bias=False, dtype=self.dtype, kernel_init=initializer)(x)

        wei = q @ rearrange(k, 'b t h -> b h t') * (self.head_size ** -0.5)
        tri = jnp.triu(jnp.full((T, T), -jnp.inf), k=1) # might need to change this
        wei = wei + tri
        wei = jax.nn.softmax(wei, axis=-1)
        wei = nn.Dropout(rate=dropout_rate)(wei, deterministic=not training)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    n_head: int
    head_size: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T, C = x.shape
        heads = [Head(self.head_size, dtype=self.dtype)(x, training=training) for _ in range(self.n_head)]
        out = jnp.concatenate(heads, axis=-1)
        out = nn.Dense(C, name='proj', dtype=self.dtype, kernel_init=initializer)(out)
        out = nn.Dropout(rate=dropout_rate)(out, deterministic=not training)
        return out

class FeedForward(nn.Module):
    n_embed: int
    n_ff: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.n_ff, dtype=self.dtype, kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_embed, dtype=self.dtype, kernel_init=initializer)(x)
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)
        return x

class Block(nn.Module):
    n_embed: int
    n_head: int
    n_ff: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = x + MultiHeadAttention(n_head=self.n_head, head_size=self.n_embed//self.n_head, name='sa_heads', dtype=self.dtype)(nn.LayerNorm(name="ln1", epsilon=1e-5)(x), training=training)
        x = x + FeedForward(n_embed=self.n_embed, n_ff=self.n_ff, name='ff', dtype=self.dtype)(nn.LayerNorm(name="ln2", epsilon=1e-5)(x), training=training)
        return x

initializer = jax.nn.initializers.normal(stddev=0.02)

class GPTLanguageModel(nn.Module):
    vocab_size: int
    n_embed: int = 32
    n_head: int = 4
    n_ff: int = 4 * n_embed
    n_layer: int = 6
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, idx: jnp.ndarray, targets: jnp.ndarray = None, training: bool = False):
        B, T = idx.shape

        tok_emb = nn.Embed(num_embeddings=self.vocab_size, name='token_embedding_table', features=self.n_embed, embedding_init=initializer)(idx) # (B, T, C)
        pos_emb = nn.Embed(num_embeddings=max_seq_len, name='position_embedding_table', features=self.n_embed, embedding_init=initializer)(jnp.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = nn.Sequential([Block(n_embed=self.n_embed, n_head=self.n_head, n_ff=self.n_ff, dtype=self.dtype) for _ in range(self.n_layer)])(x, training=training)
        x = nn.LayerNorm(name="ln_f", epsilon=1e-5)(x)
        logits = nn.Dense(self.vocab_size, name='lm_head', dtype=self.dtype, kernel_init=initializer)(x) # (B, T, vocab_size)
        logits = nn.Dropout(rate=dropout_rate)(logits, deterministic=not training)
        if targets is None:
            return logits, None
        logits = rearrange(logits, 'b t c -> (b t) c')
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets.flatten())
        return logits, jnp.mean(loss)

    def generate(self, params, key, idx, max_new_tokens: int = 100):
        # pad idx to max_new_tokens
        idx = jnp.pad(idx, ((0, 0), (0, max_new_tokens)), mode='constant', constant_values=0)

        @jax.jit
        def step_fn(i: int, idx: jnp.ndarray, subkey: jnp.ndarray):
            logits, loss = self.apply(params, idx)
            logits = logits[:, i, :] # we are not using the logits from the rest of the sequence
            token = jax.random.categorical(subkey, logits) # no need to use softmax before this
            new_idx = idx.at[:, i+1].set(token)
            return new_idx
        for i in range(max_new_tokens):
            key, subkey = jax.random.split(key)
            idx = step_fn(i, idx, key)
        return jnp.array(idx)

xb, yb = get_batch("train")

assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

m = GPTLanguageModel(vocab_size=vocab_size, n_embed=n_embed, n_head=n_head, n_ff=n_ff, n_layer=n_layer, dtype=dtype)
params = m.init(subkey, xb, yb, training=False)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("Param count:", param_count / 1e6, "M")

xb_, yb_ = get_batch("train")
out, loss = m.apply(params, xb_, yb_)

key, subkey = jax.random.split(key)
idx = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(np.array(m.generate(params, subkey, idx, max_new_tokens=500)[0])))

optimizer = optax.adamw(learning_rate, weight_decay=1e-2) # default weight decay is 1e-4, different from PyTorch
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, xb, yb):
    def loss_fn(params):
        logits, loss = m.apply(params, xb, yb, training=True, rngs={'dropout': dropout_rng})
        return loss
    
    loss, grad = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            l = val_step(params, xb, yb)
            loss += l
        out[split] = (loss / eval_iters).item()
    return out

@jax.jit
def val_step(params, xb, yb):
    _, loss = m.apply(params, xb, yb, training=False)
    return loss

start_time = timer()
# for _ in range(1000):
#     params, opt_state, loss = train_step(params, opt_state, xb, yb)
#     jax.block_until_ready(loss)
# end_time = timer()
# print("Time taken:", end_time - start_time)

for step in range(max_iters):
    xb, yb = get_batch("train")
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    
    if step % eval_interval == 0 or step == max_iters - 1:
        loss = estimate_loss()
        print(f"Step {step}, Loss: {loss}", "Elapsed time:", timer() - start_time)
        idx = jnp.zeros((1, 1), dtype=jnp.int32)

idx = jnp.zeros((1, 1), dtype=jnp.int32)
key, subkey = jax.random.split(key)
print(decode(np.array(m.generate(params, subkey, idx, max_new_tokens=500)[0])))