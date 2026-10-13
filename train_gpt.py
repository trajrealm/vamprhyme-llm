"""Train the GPT language model."""
import argparse

from models_gpt import GPT
from trainer import add_training_args, prepare, train


def add_gpt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)


def gpt_config(args: argparse.Namespace, vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
    }


def main(args: argparse.Namespace) -> None:
    tok, streams = prepare(args)
    config = gpt_config(args, tok.vocab_size)
    train(GPT(**config), "gpt", config, tok, streams, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, lr=1e-3, block_size=128, out="runs/gpt")
    add_gpt_args(parser)
    main(parser.parse_args())
