from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Value Object
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarginalResult:
    """
    Resultado de calcular la probabilidad marginal de un evento binario.

    Atributos
    ─────────
    target      : nombre de la columna objetivo (ej. 'RainTomorrow')
    p_event     : P(A)  — probabilidad de que ocurra el evento  (valor = 1)
    p_no_event  : P(¬A) — probabilidad complementaria           (valor = 0)
    n_total     : filas válidas usadas en el cálculo
    n_event     : conteo de filas donde el evento ocurre
    """
    target:     str
    p_event:    float   # P(A)
    p_no_event: float   # P(¬A)
    n_total:    int
    n_event:    int

    @property
    def n_no_event(self) -> int:
        return self.n_total - self.n_event

    def as_series(self) -> pd.Series:
        """
        Devuelve una Serie indexada por etiqueta.
        Útil para graficar directamente con matplotlib/seaborn.

        Ejemplo de uso:
            result.as_series().plot(kind='bar')
        """
        return pd.Series(
            {"P(A)": self.p_event, "P(¬A)": self.p_no_event},
            name=self.target,
        )

    def as_counts(self) -> pd.Series:
        """
        Devuelve conteos absolutos para gráficas de frecuencia.
        """
        return pd.Series(
            {"Evento (1)": self.n_event, "No evento (0)": self.n_no_event},
            name=self.target,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Calculador
# ═════════════════════════════════════════════════════════════════════════════

class MarginalProbability:
    """
    Calcula P(A) y P(¬A) para una columna binaria (0/1) de un DataFrame.

    Parámetros
    ──────────
    df     : DataFrame preprocesado (salida de DataPreprocessor)
    target : nombre de la columna binaria objetivo
    """

    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self._validate(df, target)
        self._series = df[target].dropna()
        self._target = target

    # ── API pública ───────────────────────────────────────────────────────

    def calculate(self) -> MarginalResult:
        """
        Calcula y devuelve el MarginalResult.

        Fórmulas
        ────────
            P(A)  = Σ(columna == 1) / N
            P(¬A) = 1 - P(A)
        """
        n_total  = len(self._series)
        n_event  = int(self._series.sum())
        p_event  = n_event / n_total if n_total > 0 else 0.0

        return MarginalResult(
            target     = self._target,
            p_event    = round(p_event, 6),
            p_no_event = round(1.0 - p_event, 6),
            n_total    = n_total,
            n_event    = n_event,
        )

    # ── Validación ────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame, target: str) -> None:
        if target not in df.columns:
            raise KeyError(f"La columna objetivo '{target}' no existe en el DataFrame.")
        unique = set(df[target].dropna().unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"La columna '{target}' debe ser binaria (0/1). "
                f"Valores encontrados: {unique}"
            )