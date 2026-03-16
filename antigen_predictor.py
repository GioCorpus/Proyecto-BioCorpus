"""
biocorpus/antigen_predictor.py
───────────────────────────────
Puntúa proteínas completas como candidatos antigénicos.

Combina múltiples señales:
  - Densidad de epítopos predichos
  - Propiedades fisicoquímicas globales
  - Accesibilidad superficial estimada
  - Conservación (si se proveen múltiples cepas)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from epitope_predictor import Epitope, EpitopePredictor
from utils import setup_logger


@dataclass(order=True)
class AntigenCandidate:
    final_score: float
    sequence: str            = field(compare=False)
    protein_id: str          = field(compare=False, default="")
    epitope_density: float   = field(compare=False, default=0.0)
    surface_score: float     = field(compare=False, default=0.0)
    stability_score: float   = field(compare=False, default=0.0)
    top_epitopes: list[Epitope] = field(compare=False, default_factory=list)

    def summary(self) -> str:
        return (
            f"{'─'*60}\n"
            f"Proteína    : {self.protein_id}\n"
            f"Longitud    : {len(self.sequence)} aa\n"
            f"Score final : {self.final_score:.3f}\n"
            f"  ├ Densidad epítopos : {self.epitope_density:.3f}\n"
            f"  ├ Superficie        : {self.surface_score:.3f}\n"
            f"  └ Estabilidad       : {self.stability_score:.3f}\n"
            f"Top epítopos:\n" +
            "\n".join(f"  [{e.position:4d}] {e.sequence}  score={e.score:.3f}" 
                      for e in self.top_epitopes[:5])
        )


class AntigenPredictor:
    """
    Evalúa proteínas y genera un ranking de candidatos antigénicos.

    Ejemplo:
        predictor = AntigenPredictor(config, epitope_predictor)
        candidates = predictor.rank(proteins, protein_ids)
        for c in candidates[:5]:
            print(c.summary())
    """

    # Pesos para el score final (suman 1.0)
    _WEIGHTS = {
        "epitope_density": 0.50,
        "surface_score":   0.30,
        "stability_score": 0.20,
    }

    def __init__(
        self,
        config: dict[str, Any],
        epitope_predictor: EpitopePredictor,
    ) -> None:
        self.config = config
        self.ep = epitope_predictor
        self.logger = setup_logger("AntigenPredictor", config)
        self.top_n: int = config["pipeline"]["top_candidates"]

    # ── Público ───────────────────────────────────────────────────────────────

    def rank(
        self,
        protein_sequences: list[str],
        protein_ids: list[str] | None = None,
    ) -> list[AntigenCandidate]:
        """
        Evalúa todas las proteínas y retorna los mejores candidatos.

        Args:
            protein_sequences: Lista de secuencias de aminoácidos.
            protein_ids: Identificadores opcionales (ej. nombres de genes).

        Returns:
            Lista de AntigenCandidate ordenada por score descendente.
        """
        if protein_ids is None:
            protein_ids = [f"protein_{i}" for i in range(len(protein_sequences))]

        self.logger.info(f"Evaluando {len(protein_sequences)} proteínas...")
        candidates: list[AntigenCandidate] = []

        for pid, seq in zip(protein_ids, protein_sequences):
            candidate = self._evaluate_protein(pid, seq)
            if candidate:
                candidates.append(candidate)

        candidates.sort(reverse=True)
        top = candidates[: self.top_n]
        self.logger.info(f"Top {len(top)} candidatos seleccionados.")
        return top

    # ── Evaluación por proteína ───────────────────────────────────────────────

    def _evaluate_protein(self, protein_id: str, sequence: str) -> AntigenCandidate | None:
        """Calcula todos los scores de una proteína y construye el candidato."""
        if len(sequence) < 20:
            self.logger.debug(f"{protein_id}: secuencia demasiado corta, omitida.")
            return None

        try:
            epitopes = self.ep.predict(sequence)
            epitope_density = self._epitope_density_score(epitopes, sequence)
            surface_score   = self._surface_accessibility_score(sequence)
            stability_score = self._stability_score(sequence)

            final_score = (
                self._WEIGHTS["epitope_density"] * epitope_density +
                self._WEIGHTS["surface_score"]   * surface_score   +
                self._WEIGHTS["stability_score"] * stability_score
            )

            return AntigenCandidate(
                final_score    = round(final_score, 4),
                sequence       = sequence,
                protein_id     = protein_id,
                epitope_density= round(epitope_density, 4),
                surface_score  = round(surface_score, 4),
                stability_score= round(stability_score, 4),
                top_epitopes   = epitopes[:10],
            )

        except Exception as exc:
            self.logger.error(f"Error evaluando {protein_id}: {exc}")
            return None

    # ── Scores individuales ───────────────────────────────────────────────────

    def _epitope_density_score(
        self, epitopes: list[Epitope], sequence: str
    ) -> float:
        """
        Score basado en cuántos epítopos de alta calidad cubre la proteína.
        Normalizado por la longitud de la proteína.
        """
        if not epitopes or not sequence:
            return 0.0
        covered_positions: set[int] = set()
        for ep in epitopes:
            for pos in range(ep.position, ep.position + ep.length):
                covered_positions.add(pos)
        coverage = len(covered_positions) / len(sequence)
        avg_score = np.mean([e.score for e in epitopes]) if epitopes else 0.0
        return float(0.6 * coverage + 0.4 * avg_score)

    def _surface_accessibility_score(self, sequence: str) -> float:
        """
        Estima accesibilidad superficial basado en hidrofobicidad.
        Proteínas menos hidrofóbicas tienden a tener regiones superficiales.
        """
        HYDROPHOBICITY = {
            "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
            "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
            "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
            "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
        }
        values = [HYDROPHOBICITY.get(aa, 0.0) for aa in sequence.upper()]
        if not values:
            return 0.5
        avg_hydro = np.mean(values)
        # Rango de hidrofobicidad: [-4.5, 4.5] → normalizar a [0, 1]
        # Proteínas hidrofílicas (baja hidrofobicidad) = más accesibles
        normalized = (avg_hydro - (-4.5)) / (4.5 - (-4.5))
        return float(1.0 - normalized)  # invertir: hidrofílica → mayor score

    def _stability_score(self, sequence: str) -> float:
        """
        Usa ProtParam de Biopython para estimar estabilidad termodinámica.
        Instability index < 40 → proteína estable (score alto).
        """
        try:
            analysis = ProteinAnalysis(sequence.upper())
            instability = analysis.instability_index()
            # Convertir a score [0, 1]: instabilidad < 40 es favorable
            score = max(0.0, min(1.0, 1.0 - (instability / 100.0)))
            return float(score)
        except Exception:
            return 0.5  # valor neutro si falla el análisis

