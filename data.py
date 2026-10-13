"""Random-window batches from the encoded token arrays."""
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, data_dir: Path, styles: list[str], split: str):
        arrays = [np.load(data_dir / f"{style}_{split}.npy") for style in styles]
        self.data = torch.from_numpy(np.concatenate(arrays).astype(np.int64))

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(self.data) - block_size, (batch_size,))
        x = torch.stack([self.data[i : i + block_size] for i in starts])
        y = torch.stack([self.data[i + 1 : i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)
