"""Lossless typed array layout, shared by writers and readers."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from tennis3d.datasets.schema import RallyIndex, RallyMetadata, ViewIndex, ViewMetadata
from tennis3d.synthetic.data import validate_view

# Each entry is (NumPy dtype, trailing dimensions). The first dimension varies.
PHYSICAL = {
    "t_s": ("float64", ()),
    "xyz_m": ("float64", (3,)),
    "velocity_m_s": ("float64", (3,)),
    "spin_rad_s": ("float64", (3,)),
    "label_valid": ("bool", ()),
    "segment_id": ("int64", ()),
    "prior_id": ("int64", ()),
    "segment_kind": ("int64", ()),
    "event_t_s": ("float64", ()),
    "event_code": ("int64", ()),
    "event_xyz_m": ("float64", (3,)),
    "event_precision_s": ("float64", ()),
    "hit_before": ("float64", (9,)),
    "hit_after": ("float64", (9,)),
}
VIEW = {
    **PHYSICAL,
    "ball_uv": ("float64", (2,)),
    "ball_clean_uv": ("float64", (2,)),
    "ball_candidate_uv": ("float64", (2,)),
    "ball_in_frame": ("bool", ()),
    "ball_observed": ("bool", ()),
    "ball_confidence": ("float64", ()),
    "court_uv": ("float64", (14, 2)),
    "court_clean_uv": ("float64", (14, 2)),
    "camera_K": ("float64", (3, 3)),
    "estimated_camera_K": ("float64", (3, 3)),
    "camera_R": ("float64", (3, 3)),
    "camera_t": ("float64", (3,)),
    "camera_valid": ("bool", ()),
    "image_size": ("int64", ()),
}


@dataclass
class ViewRecord:
    metadata: ViewMetadata
    arrays: dict[str, np.ndarray]


@dataclass
class RallyRecord:
    metadata: RallyMetadata
    physical: dict[str, np.ndarray]
    views: list[ViewRecord]

    def index(self) -> RallyIndex:
        return RallyIndex(
            **self.metadata.model_dump(),
            views=[
                ViewIndex(**v.metadata.model_dump(), frames=len(v.arrays["t_s"]))
                for v in self.views
            ],
        )


def array_struct(layout):
    fields = []
    for name, (dtype, shape) in layout.items():
        kind = pa.from_numpy_dtype(np.dtype(dtype))
        for size in reversed(shape):
            kind = pa.list_(pa.field("element", kind, nullable=False), size)
        fields.append(
            pa.field(
                name,
                pa.list_(pa.field("element", kind, nullable=False)),
                nullable=False,
            )
        )
    return pa.struct(fields)


def metadata_struct(model):
    fields = []
    for name in model.model_fields:
        kind = (
            pa.float64()
            if name == "physical_time_origin_s"
            else (
                pa.string()
                if name in {"rally_id", "family_id", "pool_sha256", "split"}
                else pa.int64()
            )
        )
        fields.append(pa.field(name, kind, nullable=False))
    return pa.struct(fields)


def arrow_schema(metadata: dict[bytes, bytes] | None = None) -> pa.Schema:
    view = pa.struct(
        [
            pa.field("metadata", metadata_struct(ViewMetadata), nullable=False),
            pa.field("arrays", array_struct(VIEW), nullable=False),
        ]
    )
    return pa.schema(
        [
            pa.field("metadata", metadata_struct(RallyMetadata), nullable=False),
            pa.field("physical", array_struct(PHYSICAL), nullable=False),
            pa.field(
                "views",
                pa.list_(pa.field("element", view, nullable=False)),
                nullable=False,
            ),
        ],
        metadata=metadata,
    )


def check_layout(arrays, layout):
    if set(arrays) != set(layout):
        raise ValueError(f"array keys differ: {set(arrays) ^ set(layout)}")
    for name, (dtype, shape) in layout.items():
        a = arrays[name]
        if (
            a.dtype != np.dtype(dtype)
            or a.ndim != 1 + len(shape)
            or a.shape[1:] != shape
        ):
            raise ValueError(
                f"invalid dtype/shape for {name}; expected {dtype} (*,{shape})"
            )


def validate_record(record: RallyRecord) -> None:
    RallyMetadata.model_validate(record.metadata.model_dump())
    record.index()  # Nonempty views and unique IDs.
    check_layout(record.physical, PHYSICAL)
    for view in record.views:
        ViewMetadata.model_validate(view.metadata.model_dump())
        check_layout(view.arrays, VIEW)
        validate_view(view.arrays)


def encode(record: RallyRecord) -> dict:
    validate_record(record)
    return {
        "metadata": record.metadata.model_dump(),
        "physical": {k: a.tolist() for k, a in record.physical.items()},
        "views": [
            {
                "metadata": v.metadata.model_dump(),
                "arrays": {k: a.tolist() for k, a in v.arrays.items()},
            }
            for v in record.views
        ],
    }


def _has_null(value) -> bool:
    return value is None or (
        isinstance(value, list) and any(_has_null(v) for v in value)
    )


def _decode_arrays(raw, layout):
    result = {}
    for key, (dtype, shape) in layout.items():
        # Null numeric values are forbidden, even where a mask is false.
        if _has_null(raw[key]):
            raise ValueError(f"Arrow null in numeric array {key}")
        result[key] = np.asarray(raw[key], dtype=dtype).reshape((-1, *shape))
    return result


def decode(raw: dict) -> RallyRecord:
    record = RallyRecord(
        RallyMetadata.model_validate(raw["metadata"]),
        _decode_arrays(raw["physical"], PHYSICAL),
        [
            ViewRecord(
                ViewMetadata.model_validate(v["metadata"]),
                _decode_arrays(v["arrays"], VIEW),
            )
            for v in raw["views"]
        ],
    )
    validate_record(record)
    return record


def _arrow_values(column: pa.Array, dtype, shape, key) -> np.ndarray:
    # Reject nulls at every nesting level before descending. .values ignores a
    # slice, so offsets are applied by hand (offsets/offset respect the slice).
    if column.null_count:
        raise ValueError(f"Arrow null in numeric array {key}")
    start, stop = column.offsets.to_pylist()[:2]
    column = column.values.slice(start, stop - start)
    for size in shape:
        if column.null_count:
            raise ValueError(f"Arrow null in numeric array {key}")
        if (
            not pa.types.is_fixed_size_list(column.type)
            or column.type.list_size != size
        ):
            raise ValueError(f"invalid Arrow list layout for {key}")
        column = column.values.slice(column.offset * size, len(column) * size)
    if column.null_count:
        raise ValueError(f"Arrow null in numeric array {key}")
    # np.array copies, so results own writable C-contiguous memory.
    values = np.array(column.to_numpy(zero_copy_only=False), dtype=dtype)
    return values.reshape((-1, *shape))


def _arrow_arrays(struct: pa.StructArray, layout) -> dict[str, np.ndarray]:
    if struct.null_count or len(struct) != 1:
        raise ValueError("Arrow null in numeric array struct")
    return {
        key: _arrow_values(struct.field(key), dtype, shape, key)
        for key, (dtype, shape) in layout.items()
    }


def decode_row(row: pa.Table) -> RallyRecord:
    """Decode a one-row table without building Python lists of the arrays."""
    if row.num_rows != 1:
        raise ValueError("expected exactly one row")
    cell = {name: row.column(name).combine_chunks() for name in row.column_names}
    views = cell["views"]
    if views.null_count:
        raise ValueError("Arrow null in views")
    views = views.flatten()
    if views.null_count:
        raise ValueError("Arrow null in views")
    view_metadata = views.field("metadata").to_pylist()
    view_arrays = views.field("arrays")
    if view_arrays.null_count:
        raise ValueError("Arrow null in views")
    record = RallyRecord(
        RallyMetadata.model_validate(cell["metadata"].to_pylist()[0]),
        _arrow_arrays(cell["physical"], PHYSICAL),
        [
            ViewRecord(
                ViewMetadata.model_validate(view_metadata[i]),
                _arrow_arrays(view_arrays.slice(i, 1), VIEW),
            )
            for i in range(len(views))
        ],
    )
    validate_record(record)
    return record
