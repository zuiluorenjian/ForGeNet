import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.evaluation import ManifestDataset, folder_manifest


class FolderEvaluationTests(unittest.TestCase):
    def make_pair(self, root):
        for folder in ("0_real", "1_fake"):
            directory = root / folder
            directory.mkdir(parents=True)
            Image.new("RGB", (240, 260), color=(60, 90, 120)).save(directory / "image.sample.png")

    def test_direct_class_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pair(root)
            entry, = folder_manifest(root)["datasets"]
            self.assertEqual([label for _, label in entry["samples"]], [0, 1])
            dataset = ManifestDataset(entry, {"data": root})
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset[0][0].shape), (3, 224, 224))

    def test_generator_subfolders_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("generator_b", "generator_a"):
                self.make_pair(root / name)
            (root / "generator_a" / "0_real" / "notes.txt").write_text("Not an image")
            entries = folder_manifest(root)["datasets"]
            self.assertEqual([entry["name"] for entry in entries], ["generator_a", "generator_b"])
            self.assertEqual([len(entry["samples"]) for entry in entries], [2, 2])
            for entry in entries:
                self.assertEqual(sorted(label for _, label in entry["samples"]), [0, 1])
                self.assertTrue(all((root / path).is_file() for path, _ in entry["samples"]))


if __name__ == "__main__":
    unittest.main()
