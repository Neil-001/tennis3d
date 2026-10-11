"""Synchronous CPU shard loading and checkpointed random window/corruption draws."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tennis3d.artifacts import file_sha256, json_sha256, read_json
from tennis3d.lifting.features import WORLD_SCALE_M, batch_features, build_features
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.corrupt import corrupt_observations
from tennis3d.synthetic.data import load_view, read_manifest


class ViewDataset(Dataset):
    """Checksummed synthetic-data reader, one view at a time; no simulation imports."""

    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self._descriptor_hash = file_sha256(self.path)
        self._source = None
        self._evaluation_preflight_modes: set[str] = set()
        raw = read_json(self.path)
        if "source_version" in raw:
            from tennis3d.datasets.source import V2Source

            self._source = V2Source(self.path, raw)
            self.format_version = 2
            self.records = self._source.records
            self.manifest_sha256 = self._source.spec.manifest.sha256
            self.manifest = {
                "split": self._source.spec.split,
                "config_sha256": self._source.manifest.provenance.config_sha256,
                "records": self.records,
            }
            self.identity = self._source.identity
        else:
            self.format_version = 1
            self.manifest = read_manifest(self.path)
            self.records = self.manifest["records"]
            self.manifest_sha256 = self._descriptor_hash
            self.identity = {
                "format_version": 1,
                "manifest_sha256": self.manifest_sha256,
                "order": "global-view-v1",
            }
        self.is_remote = self._source is not None and self._source.root is None
        if not self.records:
            self.close()
            raise ValueError("training/evaluation manifest must contain views")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self._source is not None:
            return self._source.read(index)
        return load_view(self.path, self.records[index], manifest=self.manifest)

    def simulation_config(self):
        raw = (
            self._source.manifest.provenance.config
            if self._source is not None
            else read_json(self.path.parent / "config.resolved.json")
        )
        if json_sha256(raw) != self.manifest["config_sha256"]:
            raise ValueError("simulation config does not match the training manifest")
        return SimulationConfig.model_validate(raw)

    def make_order(self, rng):
        if self._source is not None:
            return np.asarray(self._source.order(rng), dtype=np.int64)
        return rng.permutation(len(self))

    def verify_identity(self):
        if file_sha256(self.path) != self._descriptor_hash:
            raise ValueError("dataset descriptor/manifest changed")
        if (
            self._source is not None
            and not self.is_remote
            and file_sha256(self._source.manifest_path) != self.manifest_sha256
        ):
            raise ValueError("dataset manifest changed")
        if (
            self._source is not None
            and self._source.subset_path is not None
            and file_sha256(self._source.subset_path) != self._source.subset_file_sha256
        ):
            raise ValueError("subset descriptor changed")

    def local_files(self):
        if self.is_remote:
            raise ValueError(
                "timed trials require materialized local data; network is forbidden"
            )
        if self._source is not None:
            from tennis3d.datasets.schema import local_path

            assert self._source.root is not None
            paths = {self.path.resolve(), self._source.manifest_path.resolve()}
            if self._source.subset_path is not None:
                paths.add(self._source.subset_path.resolve())
            for si in self._source.groups:
                paths.add(
                    local_path(
                        self._source.root, self._source.manifest.shards[si].path
                    ).resolve()
                )
            return paths
        return {
            self.path.resolve(),
            (self.path.parent / "config.resolved.json").resolve(),
            *((self.path.parent / r["path"]).resolve() for r in self.records),
        }

    def close(self):
        if self._source is not None:
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WindowSampler:
    """Zero-worker loading keeps draw state exact at each completed optimizer step.

    A separate RNG owns view permutation; data_rng owns full-view corruption and
    crop selection. Physical trajectories and cameras are always read unchanged.
    """

    def __init__(self, dataset: ViewDataset, config):
        self.dataset, self.config = dataset, config
        self.corruption_config = dataset.simulation_config()
        self.rng = np.random.default_rng(config.seed + 1)
        self.data_rng = np.random.default_rng(config.seed + 2)
        self.order = dataset.make_order(self.rng)
        self.cursor = self.epoch = self.draws = self.rejected = 0

    def state_dict(self):
        return {
            "version": 2,
            "identity": dict(self.dataset.identity),
            "rng": self.rng.bit_generator.state,
            "data_rng": self.data_rng.bit_generator.state,
            "order": self.order.tolist(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "draws": self.draws,
            "rejected": self.rejected,
        }

    def load_state_dict(self, state):
        if state.get("version", 1) not in (1, 2):
            raise ValueError("unsupported sampler state version")
        if "identity" in state:
            if state["identity"] != self.dataset.identity:
                raise ValueError("checkpoint sampler dataset/order identity differs")
        elif self.dataset.format_version != 1:
            raise ValueError("legacy sampler state requires v1 data")
        for name in ("cursor", "epoch", "draws", "rejected"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("invalid sampler counter")
        if not isinstance(state["order"], list) or any(
            type(value) is not int for value in state["order"]
        ):
            raise ValueError("invalid sampler permutation")
        order = np.asarray(state["order"], dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(self.dataset))):
            raise ValueError("checkpoint sampler permutation does not match dataset")
        if not 0 <= state["cursor"] <= len(order):
            raise ValueError("invalid sampler cursor")
        if (
            state["draws"] != state["epoch"] * len(order) + state["cursor"]
            or state["rejected"] > state["draws"]
        ):
            raise ValueError("inconsistent sampler counters")
        self.rng.bit_generator.state = state["rng"]
        self.data_rng.bit_generator.state = state["data_rng"]
        self.order = order
        for name in ("cursor", "epoch", "draws", "rejected"):
            setattr(self, name, state[name])

    def _next_view(self):
        if self.cursor == len(self.order):
            self.order = self.dataset.make_order(self.rng)
            self.cursor = 0
            self.epoch += 1
        index = int(self.order[self.cursor])
        arrays = self.dataset[index]
        self.cursor += 1
        self.draws += 1
        return arrays

    def next_batch(self):
        # A failed batch restores its pre-draw state. The order array is replaced,
        # never mutated, so keeping a reference avoids copying the epoch order.
        saved = (
            self.order,
            self.cursor,
            self.epoch,
            self.draws,
            self.rejected,
            self.rng.bit_generator.state,
            self.data_rng.bit_generator.state,
        )
        try:
            return self._build_batch()
        except BaseException:
            (self.order, self.cursor, self.epoch, self.draws, self.rejected) = saved[:5]
            self.rng.bit_generator.state, self.data_rng.bit_generator.state = saved[5:]
            raise

    def _build_batch(self):
        windows, targets, labels = [], [], []
        for _ in range(self.config.batch_size):
            for _attempt in range(32):
                arrays = self._next_view()
                corrupted = corrupt_observations(
                    arrays["ball_clean_uv"],
                    arrays["t_s"],
                    arrays["image_size"],
                    self.corruption_config,
                    self.data_rng,
                )
                # Draw the saved view's complete corruption before selecting a crop.
                camera = arrays["camera_valid"]
                edges = np.diff(np.r_[False, camera, False].astype(int))
                intervals = list(
                    zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                )
                if not intervals:
                    self.rejected += 1
                    continue
                lengths = np.array([stop - start for start, stop in intervals])
                first, last = intervals[
                    int(self.data_rng.choice(len(intervals), p=lengths / lengths.sum()))
                ]
                first, last = int(first), int(last)
                upper = min(self.config.max_frames, last - first)
                lower = min(self.config.min_frames, upper)
                length = int(self.data_rng.integers(lower, upper + 1))
                start = int(self.data_rng.integers(first, last - length + 1))
                event_times = arrays["event_t_s"][np.isin(arrays["event_code"], [1, 2])]
                if (
                    len(event_times)
                    and self.data_rng.random() < self.config.event_crop_probability
                ):
                    event = int(
                        np.searchsorted(
                            arrays["t_s"], self.data_rng.choice(event_times)
                        )
                    )
                    start = int(
                        np.clip(
                            event - self.data_rng.integers(length), first, last - length
                        )
                    )
                sl = slice(start, start + length)
                observed = corrupted["ball_observed"][sl]
                if not observed.any() or not arrays["label_valid"][sl].any():
                    self.rejected += 1
                    continue
                window = build_features(
                    t_s=arrays["t_s"][sl],
                    ball_uv=corrupted["ball_uv"][sl],
                    ball_observed=observed,
                    court_uv=arrays["court_uv"][sl],
                    camera_valid=camera[sl],
                    image_size=arrays["image_size"],
                    pad_to=self.config.max_frames,
                )
                target = np.zeros((self.config.max_frames, 3), dtype=np.float32)
                target[:length] = arrays["xyz_m"][sl] / WORLD_SCALE_M
                valid = np.zeros(self.config.max_frames, dtype=bool)
                valid[:length] = arrays["label_valid"][sl]
                windows.append(window)
                targets.append(torch.from_numpy(target))
                labels.append(torch.from_numpy(valid))
                break
            else:
                raise ValueError(
                    "32 training draws had no supported labelled crop with ball evidence"
                )
        return batch_features(windows), torch.stack(targets), torch.stack(labels)


def validate_split_pair(train: ViewDataset, validation: ViewDataset):
    if train.manifest["split"] != "train" or validation.manifest["split"] != "val":
        raise ValueError(
            "training requires a train manifest and a held-out val manifest"
        )
    for key in ("rally_id", "family_id", "pool_sha256"):
        if {r[key] for r in train.records} & {r[key] for r in validation.records}:
            raise ValueError(f"training/validation share {key}")
