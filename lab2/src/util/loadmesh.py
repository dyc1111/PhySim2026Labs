import hashlib
from pathlib import Path
from typing import Any, Dict
import numpy as np
import trimesh


DEFAULT_SDF_RESOLUTION = 16

_LAB2_ROOT = Path(__file__).resolve().parents[2]
_SDF_CACHE_DIR = _LAB2_ROOT / "assets" / "sdf_cache"

_REQUIRED_CACHE_KEYS = (
    "sdf",
    "bbox_min",
    "bbox_max",
    "resolution",
    "meta_asset_key",
    "meta_asset_hash",
    "meta_scale",
    "meta_convexify",
)


def _parse_scale(size):
    if isinstance(size, (int, float, np.integer, np.floating)) and not isinstance(
        size, bool
    ):
        s = float(size)
        return np.array([s, s, s], dtype=np.float32)

    scale = np.array(size, dtype=np.float32).reshape(-1)
    if scale.size != 3:
        raise ValueError(
            f"custom.size must be a scalar or vec3, got shape {tuple(scale.shape)}"
        )
    return scale.astype(np.float32)


def _resolve_lab2_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (_LAB2_ROOT / path).resolve()


def _resolve_asset_path(file_path):
    path = _resolve_lab2_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"custom mesh asset not found: {path}")
    return path


def _canonical_asset_key(file_path):
    resolved = _resolve_asset_path(file_path)
    try:
        return resolved.relative_to(_LAB2_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _scale_key(scale_vec):
    return ",".join(f"{float(v):.8g}" for v in np.asarray(scale_vec, dtype=np.float32))


def _build_cache_metadata(file_path, scale_vec, convexify):
    asset_key = _canonical_asset_key(file_path)
    asset_hash = hashlib.sha1(asset_key.encode("utf-8")).hexdigest()
    return {
        "meta_asset_key": np.array(asset_key),
        "meta_asset_hash": np.array(asset_hash),
        "meta_scale": np.asarray(scale_vec, dtype=np.float32),
        "meta_convexify": np.array(1 if convexify else 0, dtype=np.int8),
    }


def _resolve_sdf_cache_path(file_path, scale_vec, convexify, resolution):
    asset_key = _canonical_asset_key(file_path)
    payload = (
        f"{asset_key}|{_scale_key(scale_vec)}|"
        f"{int(bool(convexify))}|{int(resolution)}"
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    stem = Path(asset_key).stem
    return _SDF_CACHE_DIR / f"{stem}_r{int(resolution)}_{digest}.npz"


def _save_sdf_cache(cache_path, sdf, bbox_min, bbox_max, resolution, metadata):
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, np.ndarray] = {
        "sdf": np.asarray(sdf, dtype=np.float32),
        "bbox_min": np.asarray(bbox_min, dtype=np.float32),
        "bbox_max": np.asarray(bbox_max, dtype=np.float32),
        "resolution": np.array(int(resolution), dtype=np.int32),
    }
    payload.update(metadata)
    np.savez_compressed(cache_file, **payload)


def _build_mesh_sdf(mesh, resolution):
    bbox_min = mesh.bounds[0].astype(np.float32)
    bbox_max = mesh.bounds[1].astype(np.float32)
    pad = 1e-4 + 0.02 * float(np.linalg.norm(bbox_max - bbox_min))
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad

    grid_x = np.linspace(float(bbox_min[0]), float(bbox_max[0]), resolution)
    grid_y = np.linspace(float(bbox_min[1]), float(bbox_max[1]), resolution)
    grid_z = np.linspace(float(bbox_min[2]), float(bbox_max[2]), resolution)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sdf = trimesh.proximity.signed_distance(mesh, points).astype(np.float32)
    sdf = -sdf
    return sdf.reshape(resolution, resolution, resolution), bbox_min, bbox_max


def load_custom_mesh(file_path, size, convexify):
    scale_vec = _parse_scale(size)
    asset_path = _resolve_asset_path(file_path)

    mesh = trimesh.load(asset_path, force="mesh")
    mesh.apply_scale(scale_vec)
    if convexify:
        mesh = mesh.convex_hull

    mesh.vertices -= mesh.center_mass
    return mesh, scale_vec, asset_path


def load_sdf_cache(mesh, file_path, scale_vec, convexify, expected_resolution):
    expected_metadata = _build_cache_metadata(file_path, scale_vec, convexify)
    cache_path = _resolve_sdf_cache_path(
        file_path, scale_vec, convexify, expected_resolution
    )
    cache_file = Path(cache_path)
    if not cache_file.exists():
        print("cache not found, creating cache...")
        sdf, bbox_min, bbox_max = _build_mesh_sdf(mesh, expected_resolution)
        _save_sdf_cache(
            cache_path, sdf, bbox_min, bbox_max, expected_resolution, expected_metadata
        )
        print(f"cache {cache_path} created")

    with np.load(cache_file, allow_pickle=False) as npz:
        missing_keys = [k for k in _REQUIRED_CACHE_KEYS if k not in npz.files]
        if missing_keys:
            raise ValueError(
                f"SDF cache missing required keys {missing_keys}: {cache_file}"
            )

        resolution = npz["resolution"].item()
        if resolution != int(expected_resolution):
            raise ValueError(
                f"SDF cache resolution mismatch: cache={resolution}, "
                f"expected={expected_resolution}"
            )

        meta_asset_key = npz["meta_asset_key"].item()
        expected_asset_key = expected_metadata["meta_asset_key"].item()
        if meta_asset_key != expected_asset_key:
            raise ValueError(
                f"SDF cache asset mismatch: cache={meta_asset_key}, "
                f"expected={expected_asset_key}"
            )

        meta_asset_hash = npz["meta_asset_hash"].item()
        expected_asset_hash = str(
            np.asarray(expected_metadata["meta_asset_hash"]).item()
        )
        if meta_asset_hash != expected_asset_hash:
            raise ValueError("SDF cache asset hash mismatch")

        meta_convexify = npz["meta_convexify"].item()
        expected_convexify = expected_metadata["meta_convexify"].item()
        if meta_convexify != expected_convexify:
            raise ValueError(
                f"SDF cache convexify mismatch: cache={meta_convexify}, "
                f"expected={expected_convexify}"
            )

        meta_scale = npz["meta_scale"]
        expected_scale = expected_metadata["meta_scale"]
        if not np.allclose(meta_scale, expected_scale, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"SDF cache scale mismatch: cache={meta_scale}, "
                f"expected={expected_scale}"
            )

        result: Dict[str, Any] = {
            "bbox_min": npz["bbox_min"],
            "bbox_max": npz["bbox_max"],
            "resolution": resolution,
            "sdf": npz["sdf"],
        }

    return result
