"""
biocorpus/epitope_predictor.py
───────────────────────────────
Predictor de epítopos con modelo de red neuronal (PyTorch).

Mejoras sobre el código original:
  - Scoring inmunológico real (no solo sliding window)
  - Modelo entrenado con propiedades fisicoquímicas + one-hot
  - Filtrado por umbral configurable
  - Soporte para múltiples alelos MHC
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import encode_peptide, setup_logger


# ── Dataclass de resultado ────────────────────────────────────────────────────
@dataclass(order=True)
class Epitope:
    score: float
    sequence: str       = field(compare=False)
    position: int       = field(compare=False)
    length: int         = field(compare=False)
    mhc_allele: str     = field(compare=False, default="")

    def __repr__(self) -> str:
        return (
            f"Epitope(pos={self.position}, seq={self.sequence}, "
            f"score={self.score:.3f}, mhc={self.mhc_allele})"
        )


# ── Red neuronal ─────────────────────────────────────────────────────────────
class ImmunoNet(nn.Module):
    """
    Red neuronal feedforward para predecir inmunogenicidad de péptidos.

    Input:  vector de tamaño (peptide_length * 24)  → one-hot + fisicoquímico
    Output: escalar en [0, 1] (score de inmunogenicidad)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),

            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Predictor principal ───────────────────────────────────────────────────────
class EpitopePredictor:
    """
    Predice epítopos inmunogénicos en secuencias proteicas.

    Uso básico:
        predictor = EpitopePredictor(config)
        predictor.load_or_train(X_train, y_train)
        epitopes = predictor.predict("MKTIIALSYIFCLVFA...")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = setup_logger("EpitopePredictor", config)

        pipe_cfg = config["pipeline"]
        model_cfg = config["model"]

        self.epitope_length: int = pipe_cfg["epitope_length"]
        self.min_score: float    = pipe_cfg["min_immunogenicity_score"]
        self.mhc_alleles: list[str] = pipe_cfg["mhc_alleles"]

        self.hidden_dim: int  = model_cfg["hidden_dim"]
        self.dropout: float   = model_cfg["dropout"]
        self.lr: float        = model_cfg["learning_rate"]
        self.epochs: int      = model_cfg["epochs"]
        self.batch_size: int  = model_cfg["batch_size"]

        self.input_dim = self.epitope_length * 24  # 20 one-hot + 4 fisicoquímico
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ImmunoNet(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        self._model_path = Path(config["paths"]["models_dir"]) / "immunonet.pt"
        self._model_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Dispositivo: {self.device} | Input dim: {self.input_dim}")

    # ── Público ───────────────────────────────────────────────────────────────

    def predict(
        self,
        protein_sequence: str,
        mhc_allele: str = "HLA-A*02:01",
    ) -> list[Epitope]:
        """
        Genera y puntúa todos los péptidos de longitud `epitope_length`
        dentro de la secuencia proteica.

        Returns:
            Lista de Epitope ordenada por score descendente (solo los que
            superan min_immunogenicity_score).
        """
        if len(protein_sequence) < self.epitope_length:
            self.logger.warning("Secuencia demasiado corta para extraer epítopos.")
            return []

        peptides, positions = self._sliding_window(protein_sequence)
        features = self._encode_batch(peptides)

        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            scores = self.model(tensor).cpu().numpy()

        epitopes = [
            Epitope(
                score=float(scores[i]),
                sequence=peptides[i],
                position=positions[i],
                length=self.epitope_length,
                mhc_allele=mhc_allele,
            )
            for i in range(len(peptides))
            if scores[i] >= self.min_score
        ]

        epitopes.sort(reverse=True)
        self.logger.info(
            f"Péptidos analizados: {len(peptides)} | "
            f"Epítopos candidatos (≥{self.min_score}): {len(epitopes)}"
        )
        return epitopes

    def predict_multi_allele(
        self, protein_sequence: str
    ) -> dict[str, list[Epitope]]:
        """Ejecuta la predicción para todos los alelos MHC configurados."""
        return {
            allele: self.predict(protein_sequence, mhc_allele=allele)
            for allele in self.mhc_alleles
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """
        Entrena ImmunoNet con datos etiquetados.

        Args:
            X_train: Array (N, input_dim) de péptidos codificados
            y_train: Array (N,) con scores reales de inmunogenicidad [0-1]

        Returns:
            Historial de entrenamiento: {"train_loss": [...], "val_loss": [...]}
        """
        self.logger.info(f"Iniciando entrenamiento — {len(X_train)} muestras.")

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                if epoch % 10 == 0:
                    self.logger.info(
                        f"Época {epoch:3d}/{self.epochs} | "
                        f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
                    )
            elif epoch % 10 == 0:
                self.logger.info(f"Época {epoch:3d}/{self.epochs} | Loss: {train_loss:.4f}")

        self.save_model()
        return history

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self._model_path)
        self.logger.info(f"Modelo guardado en: {self._model_path}")

    def load_model(self) -> bool:
        if self._model_path.exists():
            self.model.load_state_dict(
                torch.load(self._model_path, map_location=self.device)
            )
            self.model.eval()
            self.logger.info(f"Modelo cargado desde: {self._model_path}")
            return True
        self.logger.warning("No se encontró modelo guardado. Usa .train() primero.")
        return False

    def load_or_train(
        self,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
    ) -> None:
        """Carga modelo si existe, o lo entrena si se pasan datos."""
        if not self.load_model():
            if X_train is not None and y_train is not None:
                self.train(X_train, y_train)
            else:
                self.logger.warning("Sin modelo ni datos de entrenamiento — usando pesos aleatorios.")

    # ── Privado ───────────────────────────────────────────────────────────────

    def _sliding_window(self, sequence: str) -> tuple[list[str], list[int]]:
        """Genera todos los k-mers de la secuencia."""
        n = len(sequence) - self.epitope_length + 1
        peptides = [sequence[i : i + self.epitope_length] for i in range(n)]
        positions = list(range(n))
        # Filtrar péptidos con aminoácidos no estándar
        valid = [
            (p, pos) for p, pos in zip(peptides, positions)
            if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in p.upper())
        ]
        if not valid:
            return [], []
        peptides, positions = zip(*valid)
        return list(peptides), list(positions)

    def _encode_batch(self, peptides: list[str]) -> np.ndarray:
        """Codifica una lista de péptidos como matriz de features."""
        return np.array([encode_peptide(p, self.epitope_length) for p in peptides])

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        criterion: nn.Module,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            preds = self.model(X_t)
            return criterion(preds, y_t).item()

