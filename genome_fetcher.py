"""
biocorpus/genome_fetcher.py
────────────────────────────
Descarga y cachea genomas desde NCBI usando Biopython.
Incluye reintentos automáticos y validación de secuencias.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

from utils import load_config, retry, setup_logger


class GenomeFetcher:
    """
    Descarga secuencias genómicas desde NCBI con caché local.

    Ejemplo de uso:
        fetcher = GenomeFetcher(config)
        record = fetcher.fetch("NC_045512")  # SARS-CoV-2
    """

    SUPPORTED_DBS = {"nucleotide", "protein"}

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = setup_logger("GenomeFetcher", config)
        self._cache_dir = Path(config["paths"]["data_dir"]) / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        entrez_cfg = config["entrez"]
        Entrez.email = entrez_cfg["email"]
        if entrez_cfg.get("api_key"):
            Entrez.api_key = entrez_cfg["api_key"]

        self._delay = entrez_cfg.get("delay_between_requests", 0.5)
        self._max_retries = entrez_cfg.get("retries", 3)

    # ── Público ───────────────────────────────────────────────────────────────

    def fetch(
        self,
        accession_id: str,
        db: str = "nucleotide",
        force_download: bool = False,
    ) -> SeqRecord | None:
        """
        Descarga o carga desde caché un genoma por accession ID.

        Args:
            accession_id: Ej. "NC_045512" (SARS-CoV-2)
            db: "nucleotide" o "protein"
            force_download: Ignora el caché y fuerza nueva descarga

        Returns:
            SeqRecord de Biopython, o None si falla.
        """
        if db not in self.SUPPORTED_DBS:
            raise ValueError(f"Base de datos '{db}' no soportada. Usa: {self.SUPPORTED_DBS}")

        cache_path = self._cache_dir / f"{accession_id}.fasta"

        if cache_path.exists() and not force_download:
            self.logger.info(f"[CACHÉ] Cargando {accession_id} desde disco.")
            return self._load_from_cache(cache_path)

        self.logger.info(f"[NCBI] Descargando {accession_id} desde {db}...")
        record = self._download(accession_id, db)

        if record:
            self._save_to_cache(record, cache_path)
            self.logger.info(
                f"[OK] {accession_id} descargado — longitud: {len(record.seq):,} bp"
            )
        return record

    def fetch_multiple(
        self,
        accession_ids: list[str],
        db: str = "nucleotide",
    ) -> dict[str, SeqRecord]:
        """
        Descarga múltiples genomas respetando el rate limit de NCBI.

        Returns:
            Dict {accession_id: SeqRecord}
        """
        results: dict[str, SeqRecord] = {}
        total = len(accession_ids)

        for i, acc_id in enumerate(accession_ids, 1):
            self.logger.info(f"[{i}/{total}] Procesando {acc_id}...")
            record = self.fetch(acc_id, db)
            if record:
                results[acc_id] = record
            time.sleep(self._delay)

        self.logger.info(f"Descarga completada: {len(results)}/{total} exitosos.")
        return results

    def get_protein_sequences(self, genome_record: SeqRecord) -> list[str]:
        """
        Extrae y traduce CDS a secuencias proteicas desde un genoma nucleotídico.

        Returns:
            Lista de secuencias de aminoácidos como strings.
        """
        proteins: list[str] = []

        for feature in genome_record.features:
            if feature.type != "CDS":
                continue
            qualifiers = feature.qualifiers
            if "translation" in qualifiers:
                proteins.append(qualifiers["translation"][0])
            else:
                try:
                    cds_seq = feature.extract(genome_record.seq)
                    protein = str(cds_seq.translate(to_stop=True))
                    if len(protein) > 10:  # Filtrar péptidos muy cortos
                        proteins.append(protein)
                except Exception as exc:
                    self.logger.warning(f"No se pudo traducir CDS: {exc}")

        self.logger.info(f"Proteínas extraídas: {len(proteins)}")
        return proteins

    # ── Privado ───────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=2.0)
    def _download(self, accession_id: str, db: str) -> SeqRecord | None:
        """Descarga un registro de NCBI con reintentos."""
        with Entrez.efetch(
            db=db,
            id=accession_id,
            rettype="gb",    # GenBank incluye anotaciones CDS
            retmode="text",
        ) as handle:
            return SeqIO.read(handle, "genbank")

    def _save_to_cache(self, record: SeqRecord, path: Path) -> None:
        with open(path, "w") as f:
            SeqIO.write(record, f, "fasta")

    def _load_from_cache(self, path: Path) -> SeqRecord | None:
        try:
            return SeqIO.read(str(path), "fasta")
        except Exception as exc:
            self.logger.error(f"Error cargando caché: {exc}")
            return None

