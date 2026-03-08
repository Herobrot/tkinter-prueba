"""
posterior.py
============
Responsabilidad única: visualizar la probabilidad posterior calculada
por el módulo src/maths/bayes.py.

Consume exclusivamente los Value Objects BayesResult y MultiBayesResult;
no realiza cálculos propios de probabilidad.

Gráficas disponibles
────────────────────
    plot_posterior_single   → prior vs posterior para una evidencia
    plot_posterior_multi    → comparativa de posteriores para N evidencias
    plot_posterior_breakdown→ descomposición completa (prior/likelihood/marginal/posterior)
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import PALETTE, theme, new_figure, new_figure_grid
from src.maths.bayes import BayesResult, MultiBayesResult


# ═════════════════════════════════════════════════════════════════════════════
# Prior vs Posterior — una sola evidencia
# ═════════════════════════════════════════════════════════════════════════════

def plot_posterior_single(result: BayesResult) -> matplotlib.figure.Figure:
    """
    Compara P(A) vs P(A|B) para una evidencia puntual.

    Muestra:
        · Barras de prior y posterior
        · Área de probabilidad complementaria
        · Anotación numérica sobre cada barra

    Parámetro
    ─────────
    result : BayesResult devuelto por BayesTheorem.calculate()
    """
    fig, ax = new_figure(figsize=(7, 4.5))

    labels = [
        f"P({result.target})\n[prior]",
        f"P({result.target} | {result.evidence} > {result.threshold:.2f})\n[posterior]",
    ]
    values      = [result.prior,    result.posterior]
    complements = [result.prior_complement, result.posterior_complement]
    colors      = [PALETTE.blue,   PALETTE.red]

    x = np.arange(len(labels))
    width = 0.45

    # Barras apiladas: evento + complemento = 1
    bars_event = ax.bar(x, values, width,
                        color=colors, alpha=0.85,
                        label="P(evento)", zorder=3)
    ax.bar(x, complements, width, bottom=values,
           color=PALETTE.surface, alpha=0.55,
           label="P(¬evento)", zorder=3)

    # Anotaciones numéricas
    for bar, val in zip(bars_event, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.4f}",
                ha="center", va="bottom",
                color=PALETTE.text, fontsize=9, fontweight="bold")

    # Flecha de cambio (prior → posterior)
    delta = result.posterior - result.prior
    sign  = "▲" if delta >= 0 else "▼"
    ax.annotate(
        f"{sign} {abs(delta):.4f}",
        xy=(1, result.posterior), xytext=(1.35, (result.prior + result.posterior) / 2),
        arrowprops=dict(arrowstyle="->", color=PALETTE.yellow, lw=1.4),
        color=PALETTE.yellow, fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probabilidad")
    ax.legend(loc="upper right")
    theme.apply_axes(ax, title="Probabilidad Posterior — Teorema de Bayes")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Comparativa de posteriores — múltiples evidencias
# ═════════════════════════════════════════════════════════════════════════════

def plot_posterior_multi(multi: MultiBayesResult) -> matplotlib.figure.Figure:
    """
    Barras horizontales de P(A|Bᵢ) para cada variable de evidencia,
    con una línea vertical de referencia marcando P(A) (prior).

    Parámetro
    ─────────
    multi : MultiBayesResult devuelto por BayesTheorem.for_all_evidence()
    """
    if not multi.results:
        raise ValueError("MultiBayesResult no contiene resultados.")

    posteriors = multi.posteriors_series().sort_values(ascending=True)
    prior      = next(iter(multi.results.values())).prior

    fig, ax = new_figure(figsize=(9, max(4, len(posteriors) * 0.45 + 1.5)))

    colors = [
        PALETTE.red  if v > prior else PALETTE.blue
        for v in posteriors.values
    ]

    bars = ax.barh(posteriors.index, posteriors.values,
                   color=colors, alpha=0.82, zorder=3, height=0.6)

    # Línea de prior
    ax.axvline(prior, color=PALETTE.yellow, linestyle="--",
               linewidth=1.4, label=f"P({multi.target}) = {prior:.4f}", zorder=4)

    # Anotaciones en las barras
    for bar, val in zip(bars, posteriors.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=PALETTE.text, fontsize=8)

    ax.set_xlim(0, min(1.15, posteriors.values.max() * 1.2 + 0.1))
    ax.set_xlabel(f"P({multi.target} | evidencia > umbral)")
    ax.legend(loc="lower right")
    theme.apply_axes(ax, title="Probabilidad Posterior por Variable de Evidencia")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Descomposición completa de Bayes (prior / likelihood / marginal / posterior)
# ═════════════════════════════════════════════════════════════════════════════

def plot_posterior_breakdown(result: BayesResult) -> matplotlib.figure.Figure:
    """
    Panel de 4 mini-gráficas que desglosa cada término de la fórmula de Bayes:

        P(A|B) = P(B|A) · P(A) / P(B)

    Útil para explicar el cálculo en la UI educativa.
    """
    fig, axes = new_figure_grid(1, 4, figsize=(12, 3.8))

    items = [
        ("P(A)\nPrior",        result.prior,      result.prior_complement,      PALETTE.blue),
        ("P(B|A)\nLikelihood", result.likelihood,  1 - result.likelihood,        PALETTE.green),
        ("P(B)\nMarginal",     result.marginal_b,  1 - result.marginal_b,        PALETTE.lavender),
        ("P(A|B)\nPosterior",  result.posterior,   result.posterior_complement,  PALETTE.red),
    ]

    for ax, (label, val, comp, color) in zip(axes, items):
        # Donut / pie simplificado con dos barras verticales
        ax.bar([0], [val],  color=color,         alpha=0.85, width=0.5, zorder=3)
        ax.bar([0], [comp], color=PALETTE.surface, alpha=0.55,
               bottom=[val], width=0.5, zorder=3)
        ax.text(0, val + 0.03, f"{val:.4f}",
                ha="center", color=PALETTE.text,
                fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        theme.apply_axes(ax, title=label)

    fig.suptitle(
        f"Descomposición — P({result.target} | {result.evidence} > {result.threshold:.2f})",
        color=PALETTE.text, fontsize=11, fontweight="bold",
    )
    fig.tight_layout(pad=2.5)
    return fig