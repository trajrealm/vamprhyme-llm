"""Character-level tokenizer with support for multi-character special tokens."""
import json
import re
from collections.abc import Iterable
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str], specials: list[str]):
        self.specials = specials
        self.itos = specials + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self._special_re = re.compile("(" + "|".join(map(re.escape, specials)) + ")")

    @classmethod
    def from_texts(cls, texts: Iterable[str], specials: list[str]) -> "CharTokenizer":
        return cls(sorted(set("".join(texts))), specials)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["chars"], data["specials"])

    def save(self, path: Path) -> None:
        data = {"specials": self.specials, "chars": self.itos[len(self.specials):]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids = []
        for part in self._special_re.split(text):
            if part in self.specials:
                ids.append(self.stoi[part])
            else:
                ids.extend(self.stoi[c] for c in part)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)
