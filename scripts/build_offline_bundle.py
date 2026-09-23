"""Bundle prepared data, project sources and Linux tokenizer wheels without network."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    for split in ("train", "val"):
        if sha256(directory / f"{split}.bin") != metadata["splits"][split]["tokens_sha256"]:
            raise ValueError(f"{split} token file checksum mismatch")
    if sha256(directory / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = verify_data(args.data)
    wheels = list(args.wheelhouse.glob("tokenizers-0.21.4-*manylinux*x86_64.whl"))
    if not wheels:
        raise ValueError("wheelhouse needs a tokenizers 0.21.4 Linux x86_64 wheel")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="offline-", dir=args.output.parent) as temporary:
        staging = Path(temporary) / "tiny-transformer-offline"
        staging.mkdir()
        for directory in ("tiny_transformer", "configs", "tests", "scripts", "docs", "csrc"):
            shutil.copytree(ROOT / directory, staging / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "pyproject.toml", "Dockerfile", ".gitignore", ".dockerignore"):
            shutil.copy2(ROOT / name, staging / name)
        shutil.copytree(args.data, staging / "data" / args.data.name)
        (staging / "wheelhouse").mkdir()
        for wheel in wheels:
            shutil.copy2(wheel, staging / "wheelhouse" / wheel.name)
        bundle_info = {"format_version": 1, "data_path": f"data/{args.data.name}",
                       "target": "Linux x86_64; NGC pytorch:25.08-py3 (Python 3.12)",
                       "tokenizers_version": "0.21.4", "data_source": metadata["source"],
                       "includes": ["project", "prepared tokens", "tokenizer", "selected raw texts", "Linux tokenizer wheel"],
                       "not_included": ["NGC container image", "NVIDIA driver", "CUTLASS source", "trained model weights"]}
        (staging / "BUNDLE.json").write_text(json.dumps(bundle_info, indent=2) + "\n")
        paths = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = "".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in paths)
        (staging / "CHECKSUMS.sha256").write_text(manifest)
        with tarfile.open(args.output, "w:gz", compresslevel=6) as archive:
            archive.add(staging, arcname=staging.name)
    archive_hash = sha256(args.output)
    checksum = args.output.with_name(args.output.name + ".sha256")
    checksum.write_text(f"{archive_hash}  {args.output.name}\n")
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": archive_hash, "data": metadata["splits"]}, indent=2))


if __name__ == "__main__":
    main()
