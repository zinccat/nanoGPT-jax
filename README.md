# nanoGPT-jax
nanoGPT in jax

## Usage
```bash
python train.py
```

## Benchmark
|      | PyTorch | Jax   |
| ---- | ------- | ----- |
| bf16 |         | 35.8s |
| fp32 | 76.9s   | 55.7s |