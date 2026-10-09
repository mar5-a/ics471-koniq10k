"""Synthetic correctness checks; these are NOT dataset experiment results."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.mlp_data import Preprocessor, prepare_data, split_indices
from src.mlp_models import DEEP, SHALLOW, MLP
from src.mlp_training import regression_metrics, train_model, selection_key
from src.run_milestone2 import run_experiments


class PipelineTests(unittest.TestCase):
    def test_splits_and_stratification(self):
        labels = np.repeat([0, 1], 50)
        splits = split_indices(100, labels=labels)
        self.assertEqual([len(v) for v in splits.values()], [64, 16, 20])
        self.assertEqual(len(set(np.concatenate(list(splits.values())))), 100)
        for ids in splits.values():
            self.assertEqual(int(labels[ids].sum()), len(ids) // 2)
        predefined = ["training"] * 70 + ["validation"] * 10 + ["test"] * 20
        official = split_indices(100, predefined)
        np.testing.assert_array_equal(official["test"], np.arange(80, 100))
        koniq = split_indices(10073, ["training"] * 7058 + ["validation"] * 1000 + ["test"] * 2015)
        self.assertEqual([len(v) for v in koniq.values()], [6446, 1612, 2015])
        for name in splits:
            np.testing.assert_array_equal(splits[name], split_indices(100, labels=labels)[name])

    def test_training_only_normalization_and_roundtrip(self):
        train = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2) / 48
        targets = torch.tensor([10., 20., 30., 40.])
        scaler = Preprocessor.fit(train, targets)
        x, y = scaler.transform(train, targets).tensors
        torch.testing.assert_close(x.reshape(4, 3, 2, 2).mean((0, 2, 3)), torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(scaler.inverse_target(y).ravel(), targets)
        before = scaler.channel_mean.clone()
        scaler.transform(torch.ones_like(train) * 100, targets + 1000)
        torch.testing.assert_close(scaler.channel_mean, before)
        self.assertEqual(x.shape, (4, 12))

    def test_architectures_and_metrics(self):
        for widths in (SHALLOW, DEEP):
            model = MLP(12, widths)
            self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model.modules()), len(widths) + 1)
            self.assertEqual(model(torch.randn(5, 12)).shape, (5, 1))
        result = regression_metrics([1, 2, 3], [2, 3, 4])
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["rmse"], 1)
        self.assertAlmostEqual(result["srocc"], 1)
        self.assertTrue(np.isnan(regression_metrics([1, 2, 3], [2, 2, 2])["srocc"]))
        self.assertGreater(selection_key({"srocc": -1., "rmse": 100.}),
                           selection_key({"srocc": float("nan"), "rmse": 1.}))
        self.assertGreater(selection_key({"srocc": 0.5, "rmse": 1.}),
                           selection_key({"srocc": 0.5, "rmse": 2.}))

    def test_end_to_end_study_and_leakage(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            rng = np.random.default_rng(4)
            rows = []
            for i in range(50):
                name = f"{i}.png"
                Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(images / name)
                rows.append({"image_name": name, "MOS": 10 + i * 1.5})
            metadata = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            bundle = prepare_data(metadata, images, (8, 8))
            train = bundle.manifest.query("split == 'training'")
            self.assertAlmostEqual(bundle.preprocessor.target_mean, train.MOS.mean())
            # Modifying held-out pixels and labels must not change fitted stats.
            modified = pd.DataFrame(rows)
            held_out = bundle.manifest.query("split != 'training'")
            modified.loc[held_out.index, "MOS"] = 99
            modified.to_csv(metadata, index=False)
            for name in held_out.image_name:
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images / name)
            changed = prepare_data(metadata, images, (8, 8))
            torch.testing.assert_close(changed.preprocessor.channel_mean, bundle.preprocessor.channel_mean)
            torch.testing.assert_close(changed.preprocessor.channel_std, bundle.preprocessor.channel_std)
            self.assertEqual(changed.preprocessor.target_mean, bundle.preprocessor.target_mean)
            # Same seed yields the same initialization, ordering, and history.
            a = train_model(bundle, (8, 4), "mse", "adam", 1e-3, epochs=2)
            original_test = bundle.datasets["test"]
            bundle.datasets["test"] = torch.utils.data.TensorDataset(
                original_test.tensors[0] * 1000, original_test.tensors[1] + 1000)
            b = train_model(bundle, (8, 4), "mse", "adam", 1e-3, epochs=2)
            pd.testing.assert_frame_equal(a.history, b.history)
            bundle.datasets["test"] = original_test
            best = a.history.sort_values(["val_srocc", "val_rmse"], ascending=[False, True]).iloc[0]
            self.assertEqual(a.best_epoch, int(best.epoch))
            output = root / "study"
            result = run_experiments(bundle, output, epochs=2, batch_size=16)
            self.assertEqual(len(result["grid"]), 9)
            self.assertEqual(len(result["learning_rates"]), 4)
            self.assertEqual(result["grid"].status.tolist(), ["ok"] * 9)
            self.assertEqual(set(result["test"].network), {"shallow", "deep", "training-mean baseline"})
            self.assertEqual(json.loads((output / "config.json").read_text())["status"], "complete")
            predictions = pd.read_csv(output / "shallow_test_predictions.csv")
            expected = bundle.manifest.query("split == 'test'")
            self.assertEqual(predictions.image_name.tolist(), expected.image_name.tolist())
            np.testing.assert_allclose(predictions.target_mos, expected.MOS, atol=1e-5)
            for name in ("shallow", "deep"):
                checkpoint = torch.load(output / name / "checkpoint.pt", weights_only=True)
                model = MLP(checkpoint["settings"]["input_dim"], checkpoint["settings"]["hidden_dims"])
                model.load_state_dict(checkpoint["model_state"])
                self.assertTrue((output / f"{name}_curves.png").stat().st_size > 0)
            with self.assertRaises(FileExistsError):
                run_experiments(bundle, output, epochs=1)
            # Invalid input must fail rather than silently dropping samples.
            modified.loc[1, "image_name"] = modified.loc[0, "image_name"]
            modified.to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                prepare_data(metadata, images)
            modified.loc[1, "image_name"] = "absent.png"
            modified.to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "Missing 1"):
                prepare_data(metadata, images)


if __name__ == "__main__":
    unittest.main()
