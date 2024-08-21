# nanoGPT-jax
nanoGPT in jax

## Usage
On RTX 3090, 500 steps takes
```bash
python gpt.py
```

## Benchmark
|      | PyTorch | Jax   |
| ---- | ------- | ----- |
| bf16 |         | 35.8s |
| fp32 | 76.9s   | 55.7s |