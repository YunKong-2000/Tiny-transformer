"""Prepare local reproducible token files; network is used only by tinystories."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from .tokenizer import Tokenizer


def smoke_stories(count, seed):
    rng = random.Random(seed)
    names = ["Lily", "Ben", "Mia", "Leo", "Sam", "Nora"]
    objects = ["a red kite", "a little boat", "a blue ball", "a shiny stone", "a paper bird"]
    places = ["the garden", "the park", "the river", "the hill"]
    for i in range(count):
        name, thing, place = rng.choice(names), rng.choice(objects), rng.choice(places)
        yield f"Story {seed}-{i}. {name} found {thing} near {place}. A friend came to help. They played together and went home happy."


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value.get("text"), str):
                    raise ValueError("each JSONL row must contain a string field 'text'")
                yield value["text"]


def write_texts(path, texts):
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            if text.strip():
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
    if count == 0:
        raise ValueError("empty text split")
    return count


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output directory")
    if min(args.train_docs, args.val_docs, args.tokenizer_docs) <= 0:
        raise ValueError("document counts must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    try:
        source = {"name": args.source, "seed": args.seed}
        if args.source == "smoke":
            train_texts = smoke_stories(args.train_docs, args.seed)
            val_texts = smoke_stories(args.val_docs, args.seed + 1)
            source["notice"] = "Synthetic smoke fixtures, NOT TinyStories and NOT a quality benchmark."
        elif args.source == "local":
            if not args.train_jsonl or not args.val_jsonl:
                raise ValueError("local requires --train-jsonl and --val-jsonl")
            train_texts = itertools.islice(read_jsonl(args.train_jsonl), args.train_docs)
            val_texts = itertools.islice(read_jsonl(args.val_jsonl), args.val_docs)
        else:
            try:
                from datasets import load_dataset
                from huggingface_hub import HfApi
            except ImportError as error:
                raise RuntimeError("TinyStories requires: pip install -e '.[data]'") from error
            revision = HfApi().dataset_info("roneneldan/TinyStories", revision=args.revision).sha
            source.update({"dataset": "roneneldan/TinyStories", "revision": revision, "splits": ["train", "validation"]})
            train = load_dataset(source["dataset"], revision=revision, split="train", streaming=True)
            validation = load_dataset(source["dataset"], revision=revision, split="validation", streaming=True)
            train_texts = (row["text"] for row in itertools.islice(train, args.train_docs))
            val_texts = (row["text"] for row in itertools.islice(validation, args.val_docs))
        counts = {}
        for split, texts in (("train", train_texts), ("val", val_texts)):
            counts[split] = {"documents": write_texts(staging / f"{split}.jsonl", texts)}
            print(f"materialized {split}: {counts[split]['documents']} documents", flush=True)
        kind = args.tokenizer or ("byte" if args.source == "smoke" else "bpe")
        tokenizer = Tokenizer({"kind": "byte"}) if kind == "byte" else Tokenizer.train_bpe(
            itertools.islice(read_jsonl(staging / "train.jsonl"), args.tokenizer_docs), args.vocab_size)
        tokenizer.save(staging / "tokenizer.json")
        dtype = np.dtype("<u2" if tokenizer.vocab_size <= 65536 else "<u4")
        for split in ("train", "val"):
            token_count = 0
            with (staging / f"{split}.bin").open("wb") as handle:
                for text in read_jsonl(staging / f"{split}.jsonl"):
                    ids = tokenizer.encode(text, bos=True, eos=True)
                    handle.write(np.asarray(ids, dtype=dtype).tobytes())
                    token_count += len(ids)
            counts[split]["tokens"] = token_count
            counts[split]["tokens_sha256"] = digest(staging / f"{split}.bin")
            counts[split]["text_sha256"] = digest(staging / f"{split}.jsonl")
            if not args.keep_text:
                (staging / f"{split}.jsonl").unlink()
        metadata = {"format_version": 1, "source": source, "dtype": dtype.str, "vocab_size": tokenizer.vocab_size,
                    "tokenizer_sha256": digest(staging / "tokenizer.json"), "splits": counts,
                    "documents": "BOS + encoded text + EOS", "tokenizer_training_split": "train only"}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
        os.replace(staging, output)
        print(json.dumps(metadata, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["smoke", "tinystories", "local"], default="smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-docs", type=int, default=100000)
    parser.add_argument("--val-docs", type=int, default=2000)
    parser.add_argument("--tokenizer-docs", type=int, default=20000)
    parser.add_argument("--tokenizer", choices=["byte", "bpe"])
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-jsonl")
    parser.add_argument("--val-jsonl")
    parser.add_argument("--keep-text", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
