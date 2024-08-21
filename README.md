# nanoGPT-jax
nanoGPT in jax

## Usage
```bash
python gpt.py
```

## Benchmark
On RTX 3090, 500 steps takes
|      | PyTorch | Jax   |
| ---- | ------- | ----- |
| bf16 |         | 35.8s |
| fp32 | 76.9s   | 55.7s |