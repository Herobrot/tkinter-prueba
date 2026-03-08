"""
src/ui/state.py
===============
Responsabilidad única: mantener el estado mutable de la aplicación en
un único objeto centralizado (AppState).

Ningún panel ni widget guarda estado propio de la sesión;
todos leen y escriben sobre AppState.

El estado se pasa por referencia a Controller y paneles,
garantizando coherencia sin acoplamiento directo entre widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Importaciones de Value Objects de maths (solo tipos, sin cómputo)
from src.maths.marginal    import MarginalResult
from src.maths.conditional import ConditionalResult, MultiEvidenceConditional
from src.maths.bayes       import BayesResult, MultiBayesResult
from src.maths.naive_bayes import NaiveBayesResult


@dataclass
class AppState:
    """
    Estado global de la sesión de análisis.

    Se inicializa vacío y se rellena progresivamente a medida que
    el usuario carga datos y ejecuta cálculos.

    Secciones
    ─────────
    · Datos crudos     : df, schema
    · Selecciones UI   : target, evidence, threshold_pct, threshold_val
    · Resultados maths : marginal, conditional, bayes, multi_bayes, nb_result
    · Estado de sesión : loaded, computed, status_msg
    """

    # ── Datos preprocesados ───────────────────────────────────────────────
    df:     Optional[pd.DataFrame] = None
    schema: Optional[object]       = None   # WeatherSchema

    # ── Selecciones del usuario ───────────────────────────────────────────
    target:        str   = "RainTomorrow"
    evidence:      str   = "Humidity3pm"
    threshold_pct: int   = 50        # percentil seleccionado en el slider
    threshold_val: float = 0.0       # valor real calculado a partir del percentil

    # ── Resultados de cálculos ────────────────────────────────────────────
    marginal:    Optional[MarginalResult]          = None
    conditional: Optional[ConditionalResult]       = None
    multi_cond:  Optional[MultiEvidenceConditional]= None
    bayes:       Optional[BayesResult]             = None
    multi_bayes: Optional[MultiBayesResult]        = None
    nb_result:   Optional[NaiveBayesResult]        = None

    # ── Flags de sesión ───────────────────────────────────────────────────
    loaded:   bool = False   # datos cargados correctamente
    computed: bool = False   # al menos un cálculo completado

    # ── Mensaje de estado para la status bar ─────────────────────────────
    status_msg: str = "Listo — carga datos con  ▶  Calcular"

    # ── Helpers ──────────────────────────────────────────────────────────

    def reset_results(self) -> None:
        """Limpia todos los resultados de cálculo sin tocar los datos."""
        self.marginal    = None
        self.conditional = None
        self.multi_cond  = None
        self.bayes       = None
        self.multi_bayes = None
        self.nb_result   = None
        self.computed    = False

    @property
    def numeric_cols(self) -> list[str]:
        """Lista de columnas numéricas del schema cargado."""
        if self.schema is None:
            return []
        return list(self.schema.numeric_cols)

    @property
    def binary_cols(self) -> list[str]:
        """Lista de columnas binarias (candidatas a variable objetivo)."""
        if self.schema is None:
            return []
        return list(self.schema.binary_cols)

    @property
    def all_target_candidates(self) -> list[str]:
        """Columnas válidas como variable objetivo (binarias + numéricas)."""
        return self.binary_cols + self.numeric_cols