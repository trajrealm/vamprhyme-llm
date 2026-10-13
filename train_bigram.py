"""Train the bigram language model."""
import argparse

from models_bigram import BigramLanguageModel
from trainer import add_training_args, prepare, train


def main(args: argparse.Namespace) -> None:
    tok, streams = prepare(args)
    config = {"vocab_size": tok.vocab_size}
    train(BigramLanguageModel(**config), "bigram", config, tok, streams, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, lr=1e-2, block_size=64, out="runs/bigram")
    main(parser.parse_args())
