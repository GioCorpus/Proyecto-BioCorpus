"""
biocorpus/utils.py
──────────────────
Utilidades compartidas: configuración, logging y encoding de secuencias.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Aminoácidos estándar ──────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Propiedades fisicoquímicas por aminoácido: [hidrofobicidad, carga, peso, polar]
AA_PROPERTIES: dict[str, list[float]] = {
    "A": [1.8,  0.0, 89.1,  0.0], "C": [2.5,  0.0, 121.2, 0.0],
    "D": [-3.5, -1.0, 133.1, 1.0], "E": [-3.5, -1.0, 147.1, 1.0],
    "F": [2.8,  0.0, 165.2, 0.0], "G": [-0.4, 0.0, 75.0,  0.0],
    "H": [-3.2, 0.5, 155.2, 1.0], "I": [4.5,  0.0, 131.2, 0.0],
    "K": [-3.9, 1.0, 146.2, 1.0], "L": [3.8,  0.0, 131.2, 0.0],
    "M": [1.9,  0.0, 149.2, 0.0], "N": [-3.5, 0.0, 132.1, 1.0],
    "P": [-1.6, 0.0, 115.1, 0.0], "Q": [-3.5, 0.0, 146.2, 1.0],
    "R": [-4.5, 1.0, 174.2, 1.0], "S": [-0.8, 0.0, 105.1, 1.0],
    "T": [-0.7, 0.0, 119.1, 1.0], "V": [4.2,  0.0, 117.1, 0.0],
    "W": [-0.9, 0.0, 204.2, 1.0], "Y": [-1.3, 0.0, 181.2, 1.0],
}


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Carga el archivo YAML de configuración."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logger(name: str, config: dict[str, Any]) -> logging.Logger:
    """Configura un logger con salida a consola y archivo."""
    log_cfg = config.get("logging", {})
    log_dir = Path(config["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_cfg.get("level", "INFO")))

    fmt = logging.Formatter(
        log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s")
    )

    # Consola
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Archivo
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Decoradores ───────────────────────────────────────────────────────────────
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Reintenta una función en caso de excepción."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay * attempt)
            return None
        return wrapper
    return decorator


# ── Encoding de secuencias ────────────────────────────────────────────────────
def one_hot_encode(sequence: str, length: int | None = None) -> np.ndarray:
    """
    Codifica una secuencia de aminoácidos como one-hot.
    Retorna un array de forma (len, 20).
    """
    seq = sequence.upper()
    if length:
        seq = seq[:length].ljust(length, "A")  # truncar/pad
    matrix = np.zeros((len(seq), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            matrix[i, idx] = 1.0
    return matrix


def physicochemical_encode(sequence: str, length: int | None = None) -> np.ndarray:
    """
    Codifica cada aminoácido con 4 propiedades fisicoquímicas.
    Retorna un array de forma (len, 4).
    """
    seq = sequence.upper()
    if length:
        seq = seq[:length].ljust(length, "G")
    props = [AA_PROPERTIES.get(aa, [0.0, 0.0, 0.0, 0.0]) for aa in seq]
    arr = np.array(props, dtype=np.float32)
    # Normalizar cada columna
    col_max = np.abs(arr).max(axis=0, keepdims=True) + 1e-8
    return arr / col_max


def encode_peptide(sequence: str, length: int = 9) -> np.ndarray:
    """
    Combina one-hot + propiedades fisicoquímicas.
    Retorna vector aplanado de tamaño length * 24.
    """
    oh = one_hot_encode(sequence, length)        # (length, 20)
    pc = physicochemical_encode(sequence, length) # (length,  4)
    combined = np.concatenate([oh, pc], axis=1)  # (length, 24)
    return combined.flatten()                     # (length*24,)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Crea todos los directorios definidos en config.paths si no existen."""
    for key, path in config["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)

