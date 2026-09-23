"""A dependency-free byte tokenizer and an optional Hugging Face BPE adapter."""
import json
from pathlib import Path


class Tokenizer:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec["kind"]
        if self.kind == "byte":
            self.impl = None
            self.vocab_size = 259
            self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        elif self.kind == "bpe":
            try:
                from tokenizers import Tokenizer as HFTokenizer
            except ImportError as error:
                raise RuntimeError("BPE requires: pip install -e '.[data]'") from error
            self.impl = HFTokenizer.from_str(spec["json"])
            self.vocab_size = self.impl.get_vocab_size()
            self.pad_id = self.impl.token_to_id("[PAD]")
            self.bos_id = self.impl.token_to_id("[BOS]")
            self.eos_id = self.impl.token_to_id("[EOS]")
        else:
            raise ValueError(f"unknown tokenizer {self.kind}")

    def encode(self, text, bos=False, eos=False):
        ids = [byte + 3 for byte in text.encode("utf-8")] if self.impl is None else self.impl.encode(text).ids
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids):
        if self.impl is None:
            return bytes(index - 3 for index in ids if 3 <= index < 259).decode("utf-8", errors="replace")
        return self.impl.decode(ids, skip_special_tokens=True)

    def save(self, path):
        Path(path).write_text(json.dumps(self.spec, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def train_bpe(cls, texts, vocab_size):
        try:
            from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, trainers, decoders
        except ImportError as error:
            raise RuntimeError("BPE requires: pip install -e '.[data]'") from error
        tokenizer = HFTokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls({"kind": "bpe", "json": tokenizer.to_str()})
