# M4 Machine Audit

- Product: Mac mini (Mac16,10, MU9D3FN/A)
- Chip: Apple M4; P/E/total CPU cores: 4/6/10; GPU cores: 10
- Unified memory: 16 GB
- macOS: 26.5 (25F71)
- Python: 3.13.14; packages: {'mlx': '0.31.2', 'mlx-lm': '0.31.3', 'numpy': '2.4.6', 'faiss-cpu': '1.14.3', 'matplotlib': '3.11.0'}
- Model: mlx-community/Qwen2.5-1.5B-Instruct-4bit@8b403126fc14f14cfc99bb4cfa72ecbc129ea677, 4-bit
- Index: 100,000 x 384, M=32, efConstruction=80, SHA-256 4c65bde676235523dbba2f1dc78a44de3f447470d38105488586d7ca486a51f0
- Repository commit: 57705767158d2a2e8aa4ccc0134cfd4527803b4b
- MLX memory limit: 5.5 GB
- Initial pageouts: 0; swap: vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)
- Power: AC, Low Power Mode disabled (full `pmset` output is in the JSON manifest).

This campaign targets the base Apple M4, not M4 Pro. Under the r5 stabilization
handoff, CPU scaling reruns caps 1, 2, and 4; cap 4 is tested rather than assumed
safe, and K_hi is selected only from the new r5 CPU-only measurements.
