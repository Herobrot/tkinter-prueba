from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Value Objects
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConditionalResult:
    """
    Resultado de una probabilidad condicional P(A | B).

    Atributos
    ─────────
    target      : columna objetivo (A)
    evidence    : columna de evidencia (B)
    threshold   : umbral aplicado sobre la columna de evidencia
    p_a_given_b      : P(A | B)        — P(Fallo  | Variable > umbral)
    p_a_given_not_b  : P(A | ¬B)       — P(Fallo  | Variable ≤ umbral)
    p_b_given_a      : P(B | A)        — P(Variable > umbral | Fallo)
    p_b_given_not_a  : P(B | ¬A)       — P(Variable > umbral | No fallo)
    n_b              : filas donde la evidencia supera el umbral
    n_not_b          : filas donde la evidencia no supera el umbral
    """
    target:         str
    evidence:       str
    threshold:      float

    p_a_given_b:     float   # P(Fallo | Variable > umbral)
    p_a_given_not_b: float   # P(Fallo | Variable ≤ umbral)
    p_b_given_a:     float   # P(Variable > umbral | Fallo)
    p_b_given_not_a: float   # P(Variable > umbral | No Fallo)

    n_b:     int
    n_not_b: int

    def as_series_given_threshold(self) -> pd.Series:
        """
        Serie para graficar P(Fallo | X > u) vs P(Fallo | X ≤ u).
        """
        return pd.Series(
            {
                f"P({self.target} | {self.evidence} > {self.threshold:.2f})": self.p_a_given_b,
                f"P({self.target} | {self.evidence} ≤ {self.threshold:.2f})": self.p_a_given_not_b,
            }
        )

    def as_series_given_failure(self) -> pd.Series:
        """
        Serie para graficar P(X > u | Fallo) vs P(X > u | No Fallo).
        """
        return pd.Series(
            {
                f"P({self.evidence} > {self.threshold:.2f} | {self.target})":     self.p_b_given_a,
                f"P({self.evidence} > {self.threshold:.2f} | No {self.target})":  self.p_b_given_not_a,
            }
        )


@dataclass(frozen=True)
class MultiEvidenceConditional:
    """
    Colección de ConditionalResult para múltiples columnas de evidencia.
    Facilita la iteración desde la capa de presentación.

    Atributos
    ─────────
    target  : columna objetivo común a todos los resultados
    results : dict {nombre_evidencia → ConditionalResult}
    """
    target:  str
    results: dict[str, ConditionalResult]

    def evidence_names(self) -> list[str]:
        return list(self.results.keys())

    def get(self, evidence: str) -> ConditionalResult:
        return self.results[evidence]


# ═════════════════════════════════════════════════════════════════════════════
# Calculador
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalProbability:
    """
    Calcula probabilidades condicionales entre una columna binaria objetivo
    y una columna numérica de evidencia, dado un umbral.

    Parámetros
    ──────────
    df        : DataFrame preprocesado
    target    : nombre de la columna binaria (0/1)
    evidence  : nombre de la columna numérica de evidencia
    threshold : valor de corte para binarizar la evidencia (X > threshold)
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        target:    str,
        evidence:  str,
        threshold: float,
    ) -> None:
        self._validate(df, target, evidence)
        self._df        = df[[target, evidence]].dropna()
        self._target    = target
        self._evidence  = evidence
        self._threshold = threshold

    # ── API pública ───────────────────────────────────────────────────────

    def calculate(self) -> ConditionalResult:
        """
        Calcula todas las probabilidades condicionales para un umbral.

        Fórmulas
        ────────
            sea A = (target == 1),  B = (evidence > threshold)

            P(A | B)  = P(A ∩ B)  / P(B)   = |A ∩ B|  / |B|
            P(A | ¬B) = P(A ∩ ¬B) / P(¬B)  = |A ∩ ¬B| / |¬B|
            P(B | A)  = P(A ∩ B)  / P(A)   = |A ∩ B|  / |A|
            P(B | ¬A) = P(¬A ∩ B) / P(¬A)  = |¬A ∩ B| / |¬A|
        """
        df = self._df
        a_mask =  df[self._target]   == 1
        b_mask =  df[self._evidence] >  self._threshold

        n_b     = int(b_mask.sum())
        n_not_b = int((~b_mask).sum())
        n_a     = int(a_mask.sum())
        n_not_a = int((~a_mask).sum())

        n_a_and_b     = int((a_mask &  b_mask).sum())
        n_a_and_not_b = int((a_mask & ~b_mask).sum())

        return ConditionalResult(
            target    = self._target,
            evidence  = self._evidence,
            threshold = self._threshold,

            p_a_given_b     = self._safe_div(n_a_and_b,     n_b),
            p_a_given_not_b = self._safe_div(n_a_and_not_b, n_not_b),
            p_b_given_a     = self._safe_div(n_a_and_b,     n_a),
            p_b_given_not_a = self._safe_div(n_a_and_b - n_a_and_b + int((~a_mask & b_mask).sum()), n_not_a),

            n_b     = n_b,
            n_not_b = n_not_b,
        )

    @classmethod
    def for_all_numeric(
        cls,
        df:             pd.DataFrame,
        target:         str,
        numeric_cols:   list[str],
        threshold_fn:   callable = None,
    ) -> MultiEvidenceConditional:
        """
        Calcula ConditionalResult para cada columna numérica de la lista.

        threshold_fn : función (pd.Series) → float que determina el umbral.
                       Por defecto usa la mediana de cada columna.

        Devuelve MultiEvidenceConditional con todos los resultados.
        """
        if threshold_fn is None:
            threshold_fn = lambda s: float(s.median())

        results = {}
        for col in numeric_cols:
            if col == target or col not in df.columns:
                continue
            threshold = threshold_fn(df[col].dropna())
            calc = cls(df, target, col, threshold)
            results[col] = calc.calculate()

        return MultiEvidenceConditional(target=target, results=results)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_div(numerator: int, denominator: int) -> float:
        """División segura: devuelve 0.0 si el denominador es cero."""
        return round(numerator / denominator, 6) if denominator > 0 else 0.0

    @staticmethod
    def _validate(df: pd.DataFrame, target: str, evidence: str) -> None:
        for col in (target, evidence):
            if col not in df.columns:
                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
        unique = set(df[target].dropna().unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"'{target}' debe ser binaria (0/1). Valores: {unique}"
            )