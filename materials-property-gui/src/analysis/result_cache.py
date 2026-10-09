import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from analysis.elastic_properties import (
        DATA_DIRECTORY,
        _normalize_material,
        _normalize_method,
        _normalize_structure,
        mace_model_for_method,
    )
except ModuleNotFoundError as error:
    if error.name != "analysis":
        raise
    from src.analysis.elastic_properties import (
        DATA_DIRECTORY,
        _normalize_material,
        _normalize_method,
        _normalize_structure,
        mace_model_for_method,
    )


CACHE_VERSION = 1
MEAM_ENVIRONMENT = (
    "MEAM_LIBRARY",
    "MEAM_PARAMETER_FILE",
    "MEAM_PAIR_COEFF",
    "MEAM_LIBRARY_ELEMENT",
    "LAMMPS_COMMAND",
)
def _cache_directory():
    configured_directory = os.getenv("MATERIALS_PROPERTY_CACHE")
    if configured_directory:
        return Path(configured_directory).expanduser()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "calculations"
    if os.name == "nt":
        base_directory = Path(os.getenv("LOCALAPPDATA", Path.home()))
    else:
        base_directory = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base_directory / "MaterialsPropertyGUI" / "calculations"


def _file_signature(path):
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "size": stat.st_size, "modified": stat.st_mtime_ns}


def _source_signatures(method, symbol):
    potential_directory = DATA_DIRECTORY / "potentials"
    if method == "MEAM":
        paths = list(potential_directory.glob("*.meam"))
        configured_paths = [os.getenv(name) for name in MEAM_ENVIRONMENT[:2]]
    elif method == "DFT":
        return []
    else:
        return []

    for configured_path in configured_paths:
        if configured_path:
            path = Path(configured_path).expanduser()
            if path.is_file():
                paths.append(path)
    signatures = [_file_signature(path) for path in paths]
    return sorted(
        (signature for signature in signatures if signature is not None),
        key=lambda signature: signature["path"],
    )


def _cache_identity(material, lattice_structure, method):
    material_name, symbol = _normalize_material(material)
    method_name = _normalize_method(method)
    model = mace_model_for_method(method) if method_name == "MACE-MP" else None
    environment_names = MEAM_ENVIRONMENT if method_name == "MEAM" else ()
    backend = None
    if method_name == "DFT":
        try:
            from dft.vasp_workflow import VASP_CACHE_BACKEND
        except ModuleNotFoundError as error:
            if error.name != "dft":
                raise
            from src.dft.vasp_workflow import VASP_CACHE_BACKEND
        backend = VASP_CACHE_BACKEND
    identity = {
        "version": CACHE_VERSION,
        "material": material_name,
        "symbol": symbol,
        "lattice": _normalize_structure(lattice_structure),
        "method": method_name,
        "model": model,
        "backend": backend,
        "environment": {name: os.getenv(name) for name in environment_names},
        "sources": _source_signatures(method_name, symbol),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_value(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot cache value of type {type(value).__name__}")


def _decode_value(value):
    if set(value) == {"__ndarray__"}:
        return np.asarray(value["__ndarray__"])
    return value


class CalculationResultCache:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else _cache_directory()

    def load(self, material, lattice_structure, method):
        key = _cache_identity(material, lattice_structure, method)
        cache_file = self.directory / f"{key}.json"
        try:
            with cache_file.open("r", encoding="utf-8") as file:
                entry = json.load(file, object_hook=_decode_value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(entry, dict):
            return None
        if entry.get("key") != key or not isinstance(entry.get("result"), dict):
            return None
        return entry["result"]

    def save(self, material, lattice_structure, method, result):
        key = _cache_identity(material, lattice_structure, method)
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = json.dumps(
            {"key": key, "result": result},
            default=_encode_value,
            separators=(",", ":"),
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False
            ) as file:
                temporary_path = Path(file.name)
                file.write(entry)
            temporary_path.replace(self.directory / f"{key}.json")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)