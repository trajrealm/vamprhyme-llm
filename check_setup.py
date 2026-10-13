import time, torch, numpy as np

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("CPU threads:", torch.get_num_threads())

x = torch.randn(2000, 2000)
t = time.time()
for _ in range(10):
    y = x @ x
print(f"10 matrix multiplies: {time.time() - t:.2f} s")