from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .marginal    import MarginalResult
from .conditional import ConditionalResult


# ═════════════════════════════════════════════════════════════════════════════
# Value Object
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BayesResult:
    """
    Resultado completo de aplicar el Teorema de Bayes.

    Atributos
    ─────────
    target       : columna objetivo (A)
    evidence     : columna de evidencia (B)
    threshold    : umbral sobre la evidencia

    prior          : P(A)              — probabilidad a priori
    likelihood     : P(B | A)          — verosimilitud
    marginal_b     : P(B)              — probabilidad marginal de la evidencia
    posterior      : P(A | B)          — probabilidad posterior  ← resultado principal

    prior_complement    : P(¬A)
    likelihood_complement: P(B | ¬A)
    posterior_complement : P(¬A | B)   — suma con posterior = 1
    """
    target:    str
    evidence:  str
    threshold: float

    prior:                float   # P(A)
    likelihood:           float   # P(B | A)
    marginal_b:           float   # P(B)
    posterior:            float   # P(A | B)  ← Bayes

    prior_complement:     float   # P(¬A)
    likelihood_complement: float  # P(B | ¬A)
    posterior_complement: float   # P(¬A | B)

    def as_comparison_series(self) -> pd.Series:
        """
        Serie para comparar prior vs posterior en una gráfica de barras.

        Ejemplo de uso:
            result.as_comparison_series().plot(kind='bar', color=['steelblue', 'salmon'])
        """
        return pd.Series(
            {
                f"P({self.target})":                         self.prior,
                f"P({self.target} | {self.evidence} > {self.threshold:.2f})": self.posterior,
            },
            name="Bayes",
        )

    def as_full_series(self) -> pd.Series:
        """
        Serie extendida con prior, likelihood, marginal y posterior.
        Útil para tablas detalladas en la UI.
        """
        return pd.Series(
            {
                "P(A) — prior":        self.prior,
                "P(B | A) — likelihood": self.likelihood,
                "P(B) — marginal":     self.marginal_b,
                "P(A | B) — posterior": self.posterior,
            }
        )


@dataclass(frozen=True)
class MultiBayesResult:
    """
    Colección de BayesResult para múltiples columnas de evidencia.
    Facilita la iteración y construcción de gráficas comparativas.
    """
    target:  str
    results: dict[str, BayesResult]   # {nombre_evidencia → BayesResult}

    def posteriors_series(self) -> pd.Series:
        """
        Serie {nombre_evidencia → posterior} para graficar todas las
        probabilidades posteriores en una misma gráfica.
        """
        return pd.Series(
            {col: r.posterior for col, r in self.results.items()},
            name=f"P({self.target} | evidencia)",
        )

    def priors_series(self) -> pd.Series:
        """Serie con el prior (constante) para línea de referencia."""
        if not self.results:
            return pd.Series(dtype=float)
        prior = next(iter(self.results.values())).prior
        return pd.Series(
            dict.fromkeys(self.results, prior),
            name=f"P({self.target})",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Calculador
# ═════════════════════════════════════════════════════════════════════════════

class BayesTheorem:
    """
    Aplica el Teorema de Bayes combinando un MarginalResult
    con un ConditionalResult.

    Parámetros
    ──────────
    marginal    : MarginalResult  — contiene P(A) y P(¬A)
    conditional : ConditionalResult — contiene P(B|A) y P(B|¬A)
    """

    def __init__(
        self,
        marginal:    MarginalResult,
        conditional: ConditionalResult,
    ) -> None:
        if marginal.target != conditional.target:
            raise ValueError(
                f"El target del marginal ('{marginal.target}') no coincide "
                f"con el del condicional ('{conditional.target}')."
            )
        self._marginal    = marginal
        self._conditional = conditional

    # ── API pública ───────────────────────────────────────────────────────

    def calculate(self) -> BayesResult:
        """
        Aplica la regla de Bayes.

        Fórmula
        ───────
            P(A | B) = P(B | A) · P(A) / P(B)

            donde P(B) se calcula por la ley de la probabilidad total:
            P(B) = P(B | A) · P(A)  +  P(B | ¬A) · P(¬A)
        """
        p_a   = self._marginal.p_event       # P(A)
        p_na  = self._marginal.p_no_event    # P(¬A)
        p_b_a = self._conditional.p_b_given_a      # P(B | A)
        p_b_na= self._conditional.p_b_given_not_a  # P(B | ¬A)

        # Probabilidad marginal de la evidencia — ley de probabilidad total
        p_b = p_b_a * p_a + p_b_na * p_na

        # Probabilidades posteriores
        posterior          = self._safe_div(p_b_a  * p_a,  p_b)
        posterior_complement = self._safe_div(p_b_na * p_na, p_b)

        return BayesResult(
            target    = self._marginal.target,
            evidence  = self._conditional.evidence,
            threshold = self._conditional.threshold,

            prior                = round(p_a,   6),
            likelihood           = round(p_b_a, 6),
            marginal_b           = round(p_b,   6),
            posterior            = round(posterior, 6),

            prior_complement      = round(p_na,   6),
            likelihood_complement = round(p_b_na, 6),
            posterior_complement  = round(posterior_complement, 6),
        )

    @classmethod
    def for_all_evidence(
        cls,
        marginal:    MarginalResult,
        conditionals: dict[str, ConditionalResult],
    ) -> MultiBayesResult:
        """
        Aplica Bayes para cada ConditionalResult de un conjunto de evidencias.

        Parámetros
        ──────────
        marginal     : resultado marginal único del evento objetivo
        conditionals : dict {nombre_evidencia → ConditionalResult}

        Devuelve MultiBayesResult con todos los posteriores calculados.
        """
        results = {}
        for col, cond in conditionals.items():
            calculator     = cls(marginal, cond)
            results[col]   = calculator.calculate()

        return MultiBayesResult(target=marginal.target, results=results)

    # ── Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0.0 else 0.0