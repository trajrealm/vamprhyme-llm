"""Random-window batches from the encoded token arrays, with weighted style mixing."""
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, data_dir: Path, styles: list[str], split: str, weights: list[float] | None = None):
        self.parts = [torch.from_numpy(np.load(data_dir / f"{s}_{split}.npy").astype(np.int64)) for s in styles]
        sizes = [float(len(p)) for p in self.parts]
        self.weights = torch.tensor(weights if weights is not None else sizes)

    def __len__(self) -> int:
        return sum(len(p) for p in self.parts)

    @property
    def mix(self) -> list[float]:
        return (self.weights / self.weights.sum()).tolist()

    def batch(self, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        choices = torch.multinomial(self.weights, batch_size, replacement=True).tolist()
        xs, ys = [], []
        for c in choices:
            data = self.parts[c]
            i = torch.randint(len(data) - block_size, (1,)).item()
            xs.append(data[i : i + block_size])
            ys.append(data[i + 1 : i + block_size + 1])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)
