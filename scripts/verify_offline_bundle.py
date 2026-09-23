"""Check every bundled file using Python's standard library (no network or pip)."""
import hashlib
from pathlib import Path


def verify(root):
    root = root.resolve()
    count = 0
    for line in (root / "CHECKSUMS.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError(f"invalid manifest path: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
        count += 1
    print(f"PASS: {count} files match CHECKSUMS.sha256")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
