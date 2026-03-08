"""
src/ui/controller.py
====================
Responsabilidad única: orquestar los módulos src/maths y src/graphs
a partir del AppState, y devolver figuras y resultados listos para
ser mostrados por los paneles de la UI.

El controller NO importa ningún widget de Tkinter.
Recibe AppState, llama a los calculadores y graficadores, y
actualiza el estado con los resultados.

Este diseño aplica el principio D: la UI depende de la abstracción
del controller, no de los módulos de maths/graphs directamente.
"""

from __future__ import annotations

import numpy as np
import matplotlib.figure

from src.util.init import DataPreprocessor, CsvDataLoader, WEATHER_SCHEMA

from src.maths.init import (
    MarginalProbability,
    ConditionalProbability,
    BayesTheorem,
    NaiveBayesClassifier,
)
from src.graphs.init import (
    plot_histograms,
    plot_temporal,
    plot_posterior_single,
    plot_posterior_multi,
    plot_posterior_breakdown,
    plot_confusion_panel,
    plot_prior_vs_posterior,
    plot_posterior_lift,
    plot_comparison_heatmap,
)

from .state import AppState


class BayesController:
    """
    Controlador principal de la aplicación.

    Métodos públicos
    ────────────────
    load_data()     → carga el CSV y rellena state.df / state.schema
    compute()       → ejecuta todos los cálculos sobre state, actualiza state
    fig_histograms()         → Figure histogramas
    fig_temporal()           → Figure serie temporal
    fig_posterior_single()   → Figure prior vs posterior (una evidencia)
    fig_posterior_multi()    → Figure posteriores multi-evidencia
    fig_posterior_breakdown()→ Figure descomposición de Bayes
    fig_confusion_panel()    → Figure matriz + métricas + distribución prob
    fig_prior_vs_posterior() → Figure comparación barras agrupadas
    fig_lift()               → Figure lift bayesiano
    fig_heatmap()            → Figure heatmap comparativo
    build_results_lines()    → lista de (texto, tag) para ResultsTable
    """

    def __init__(self, state: AppState) -> None:
        self._state = state

    # ── Carga de datos ────────────────────────────────────────────────────

    def load_data(self, filepath: str) -> None:
        """
        Carga y preprocesa el CSV indicado por el usuario.
        Actualiza state.df, state.schema, state.loaded.
        Lanza FileNotFoundError si el CSV no existe.
        """
        loader = CsvDataLoader(filepath)
        result = DataPreprocessor(loader=loader).run()

        self._state.df     = result.df
        self._state.schema = result.schema
        self._state.loaded = True
        self._state.status_msg = (
            f"✔  {len(result.df)} registros cargados  ·  "
            f"{len(result.df.columns)} columnas  ·  "
            f"desde {result.df['Date'].min().date()} "
            f"hasta {result.df['Date'].max().date()}"
        )

    # ── Cómputo ───────────────────────────────────────────────────────────

    def compute(self) -> None:
        """
        Ejecuta el pipeline completo de cálculo bayesiano sobre el estado.

        Orden:
          1. Probabilidad marginal P(target)
          2. Probabilidad condicional P(target | evidence > threshold)
          3. Condicionales para todas las columnas numéricas
          4. Teorema de Bayes — evidencia seleccionada
          5. Teorema de Bayes — todas las evidencias
          6. Clasificador Naive Bayes

        Actualiza todos los campos de resultados en AppState.
        """
        s  = self._state
        df = s.df

        # Umbral real a partir del percentil seleccionado
        s.threshold_val = float(
            np.percentile(df[s.evidence].dropna(), s.threshold_pct)
        )

        # 1. Marginal
        s.marginal = MarginalProbability(df, s.target).calculate()

        # 2. Condicional — evidencia seleccionada
        s.conditional = ConditionalProbability(
            df, s.target, s.evidence, s.threshold_val
        ).calculate()

        # 3. Condicionales — todas las numéricas (usando mediana como umbral)
        s.multi_cond = ConditionalProbability.for_all_numeric(
            df, s.target, s.numeric_cols,
            threshold_fn=lambda col: float(col.median()),
        )

        # 4. Bayes — evidencia seleccionada
        s.bayes = BayesTheorem(s.marginal, s.conditional).calculate()

        # 5. Bayes — todas las evidencias
        s.multi_bayes = BayesTheorem.for_all_evidence(
            s.marginal, s.multi_cond.results
        )

        # 6. Naive Bayes — solo columnas numéricas sin el target
        features = [c for c in s.numeric_cols if c != s.target]
        s.nb_result = NaiveBayesClassifier(df, s.target, features).run()

        s.computed     = True
        s.status_msg   = (
            f"✔  Calculado  ·  target={s.target}  ·  "
            f"evidencia={s.evidence}  ·  umbral={s.threshold_val:.2f}  "
            f"(p{s.threshold_pct})  ·  "
            f"P({s.target})={s.marginal.p_event:.4f}  →  "
            f"P({s.target}|Ev)={s.bayes.posterior:.4f}"
        )

    # ── Figuras ───────────────────────────────────────────────────────────

    def fig_histograms(self) -> matplotlib.figure.Figure:
        s = self._state
        return plot_histograms(s.df, s.numeric_cols, target=s.target)

    def fig_temporal(self) -> matplotlib.figure.Figure:
        s = self._state
        return plot_temporal(
            s.df, s.evidence,
            threshold=s.threshold_val,
            target=s.target,
        )

    def fig_posterior_single(self) -> matplotlib.figure.Figure:
        return plot_posterior_single(self._state.bayes)

    def fig_posterior_multi(self) -> matplotlib.figure.Figure:
        return plot_posterior_multi(self._state.multi_bayes)

    def fig_posterior_breakdown(self) -> matplotlib.figure.Figure:
        return plot_posterior_breakdown(self._state.bayes)

    def fig_confusion_panel(self) -> matplotlib.figure.Figure:
        return plot_confusion_panel(self._state.nb_result)

    def fig_prior_vs_posterior(self) -> matplotlib.figure.Figure:
        return plot_prior_vs_posterior(self._state.multi_bayes)

    def fig_lift(self) -> matplotlib.figure.Figure:
        return plot_posterior_lift(self._state.multi_bayes)

    def fig_heatmap(self) -> matplotlib.figure.Figure:
        return plot_comparison_heatmap(self._state.multi_bayes)

    # ── Datos para ResultsTable ───────────────────────────────────────────

    def build_results_lines(self) -> list[tuple[str, str]]:
        """
        Construye la lista de (texto, tag) para ResultsTable.write().
        tags: "header" | "value" | "high" | "neutral" | ""
        """
        s = self._state
        if not s.computed:
            return [("Sin resultados. Pulsa ▶ Calcular.", "neutral")]

        m  = s.marginal
        b  = s.bayes
        nb = s.nb_result
        cm = nb.metrics.confusion

        lines: list[tuple[str, str]] = []

        # — Marginal ──────────────────────────────────────────────────────
        lines += [
            ("── Probabilidad Marginal ───────────────────", "header"),
            (f"  P({m.target})            = {m.p_event:.6f}", "value"),
            (f"  P(¬{m.target})           = {m.p_no_event:.6f}", "neutral"),
            (f"  N total = {m.n_total}  |  N evento = {m.n_event}", "neutral"),
            ("", ""),
        ]

        # — Condicional ───────────────────────────────────────────────────
        c = s.conditional
        lines += [
            ("── Probabilidad Condicional ────────────────", "header"),
            (f"  P({c.target} | {c.evidence} > {c.threshold:.2f})   = {c.p_a_given_b:.6f}", "value"),
            (f"  P({c.target} | {c.evidence} ≤ {c.threshold:.2f})   = {c.p_a_given_not_b:.6f}", "neutral"),
            (f"  P({c.evidence} > {c.threshold:.2f} | {c.target})   = {c.p_b_given_a:.6f}", "value"),
            (f"  P({c.evidence} > {c.threshold:.2f} | ¬{c.target})  = {c.p_b_given_not_a:.6f}", "neutral"),
            ("", ""),
        ]

        # — Bayes ─────────────────────────────────────────────────────────
        lines += [
            ("── Teorema de Bayes ────────────────────────", "header"),
            (f"  P(A)  prior              = {b.prior:.6f}", "neutral"),
            (f"  P(B|A) likelihood        = {b.likelihood:.6f}", "neutral"),
            (f"  P(B)  marginal           = {b.marginal_b:.6f}", "neutral"),
            (f"  P(A|B) posterior  ◀      = {b.posterior:.6f}", "high"),
            (f"  Δ posterior − prior      = {b.posterior - b.prior:+.6f}", "value"),
            ("", ""),
        ]

        # — Naive Bayes ───────────────────────────────────────────────────
        met = nb.metrics
        lines += [
            ("── Clasificador Naive Bayes ────────────────", "header"),
            (f"  Features usadas          : {len(nb.features)}", "neutral"),
            (f"  Accuracy                 = {met.accuracy:.6f}", "value"),
            (f"  Sensibilidad (Recall)    = {met.sensitivity:.6f}", "value"),
            (f"  Especificidad            = {met.specificity:.6f}", "value"),
            (f"  Precisión                = {met.precision:.6f}", "value"),
            ("", ""),
            ("── Matriz de Confusión ─────────────────────", "header"),
            (f"  TP={cm.tp}  FP={cm.fp}  TN={cm.tn}  FN={cm.fn}", "neutral"),
        ]

        return lines