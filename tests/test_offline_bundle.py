import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.verify_offline_bundle import verify


class OfflineBundleTests(unittest.TestCase):
    def test_verification_detects_corrupted_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"offline data\x00\x01"
            (root / "tokens.bin").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (root / "CHECKSUMS.sha256").write_text(f"{digest}  tokens.bin\n")
            verify(root)
            (root / "tokens.bin").write_bytes(payload + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify(root)

    def test_manifest_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CHECKSUMS.sha256").write_text("0" * 64 + "  ../outside.bin\n")
            with self.assertRaisesRegex(ValueError, "invalid manifest path"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
