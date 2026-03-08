"""
comparison.py
=============
Responsabilidad única: comparar visualmente P(A) (prior) contra
P(A|B) (posterior) para una o múltiples variables de evidencia.

Consume exclusivamente Value Objects de src/maths:
    · MarginalResult
    · BayesResult
    · MultiBayesResult

No realiza ningún cálculo de probabilidad.

Gráficas disponibles
────────────────────
    plot_prior_vs_posterior        → barras agrupadas prior/posterior por evidencia
    plot_posterior_lift            → "lift" = posterior / prior por variable
    plot_comparison_heatmap        → heatmap de prior vs posterior (evidencias × probabilidad)
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import PALETTE, theme, new_figure, new_figure_grid
from src.maths.marginal import MarginalResult
from src.maths.bayes    import BayesResult, MultiBayesResult


# ═════════════════════════════════════════════════════════════════════════════
# Barras agrupadas: P(Fallo) vs P(Fallo|Evidencia) para N evidencias
# ═════════════════════════════════════════════════════════════════════════════

def plot_prior_vs_posterior(
    multi:    MultiBayesResult,
    max_cols: int = 10,
) -> matplotlib.figure.Figure:
    """
    Barras agrupadas que muestran, para cada variable de evidencia:
        · P(A)        — prior  (constante, barra azul)
        · P(A|B > u)  — posterior (barra roja o verde según suba/baje)

    Parámetros
    ──────────
    multi    : MultiBayesResult de BayesTheorem.for_all_evidence()
    max_cols : número máximo de evidencias a mostrar (las de mayor delta)

    Retorna
    ───────
    matplotlib.figure.Figure
    """
    if not multi.results:
        raise ValueError("MultiBayesResult no contiene resultados.")

    # Ordenar por |posterior - prior| descendente y limitar
    prior      = next(iter(multi.results.values())).prior
    sorted_res = sorted(
        multi.results.items(),
        key=lambda kv: abs(kv[1].posterior - prior),
        reverse=True,
    )[:max_cols]

    evidences  = [kv[0] for kv in sorted_res]
    posteriors = [kv[1].posterior for kv in sorted_res]
    n          = len(evidences)

    fig, ax = new_figure(figsize=(max(8, n * 0.9 + 2), 5))

    x     = np.arange(n)
    width = 0.35

    # Barras de prior (constante)
    ax.bar(x - width / 2, [prior] * n, width,
           color=PALETTE.blue, alpha=0.80,
           label=f"P({multi.target}) = {prior:.4f}",
           zorder=3)

    # Barras de posterior (color según dirección del cambio)
    post_colors = [
        PALETTE.red   if p > prior else PALETTE.teal
        for p in posteriors
    ]
    bars_post = ax.bar(x + width / 2, posteriors, width,
                       color=post_colors, alpha=0.80,
                       label=f"P({multi.target} | evidencia > umbral)",
                       zorder=3)

    # Anotaciones encima de cada barra posterior
    for bar, val in zip(bars_post, posteriors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom",
            color=PALETTE.text, fontsize=7, fontweight="bold",
        )

    # Línea de prior como referencia continua
    ax.axhline(prior, color=PALETTE.yellow, linestyle="--",
               linewidth=1.2, alpha=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(evidences, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, min(1.15, max(posteriors + [prior]) * 1.3 + 0.05))
    ax.set_ylabel("Probabilidad")
    ax.legend(loc="upper right", fontsize=8)
    theme.apply_axes(ax, title=f"P({multi.target}) vs P({multi.target} | Evidencia)")
    fig.tight_layout(pad=2.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Lift: cuánto cambia la probabilidad respecto al prior
# ═════════════════════════════════════════════════════════════════════════════

def plot_posterior_lift(multi: MultiBayesResult) -> matplotlib.figure.Figure:
    """
    Gráfica de barras horizontales del "lift bayesiano" por variable:

        lift(Bᵢ) = P(A|Bᵢ) / P(A)

    lift > 1 → la evidencia aumenta la probabilidad del evento
    lift < 1 → la evidencia la reduce
    lift = 1 → la evidencia es irrelevante (línea de referencia)

    Parámetro
    ─────────
    multi : MultiBayesResult
    """
    if not multi.results:
        raise ValueError("MultiBayesResult no contiene resultados.")

    prior = next(iter(multi.results.values())).prior
    lifts = pd.Series(
        {col: (r.posterior / prior if prior > 0 else 0.0)
         for col, r in multi.results.items()},
        name="Lift",
    ).sort_values(ascending=True)

    fig, ax = new_figure(figsize=(9, max(4, len(lifts) * 0.45 + 1.5)))

    colors = [PALETTE.red if v > 1 else PALETTE.teal for v in lifts.values]
    bars   = ax.barh(lifts.index, lifts.values,
                     color=colors, alpha=0.82, height=0.6, zorder=3)

    # Línea de referencia lift = 1
    ax.axvline(1.0, color=PALETTE.yellow, linestyle="--",
               linewidth=1.4, label="Lift = 1 (sin efecto)", zorder=4)

    for bar, val in zip(bars, lifts.values):
        ax.text(val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}×", va="center",
                color=PALETTE.text, fontsize=8)

    ax.set_xlabel(f"Lift = P({multi.target}|Bᵢ) / P({multi.target})")
    ax.legend(loc="lower right")
    theme.apply_axes(ax, title=f"Lift Bayesiano por Variable de Evidencia — {multi.target}")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Heatmap: prior vs posterior para todas las evidencias
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison_heatmap(multi: MultiBayesResult) -> matplotlib.figure.Figure:
    """
    Heatmap donde:
        · Filas    = variables de evidencia
        · Columnas = [P(A), P(A|B), Δ]
        · Celda    = valor numérico

    Permite detectar de un vistazo qué evidencias desplazan más
    la probabilidad del evento.

    Parámetro
    ─────────
    multi : MultiBayesResult
    """
    if not multi.results:
        raise ValueError("MultiBayesResult no contiene resultados.")

    prior   = next(iter(multi.results.values())).prior
    rows    = []
    indices = []

    for col, res in multi.results.items():
        delta = res.posterior - res.prior
        rows.append({
            "P(A) prior":     res.prior,
            "P(A|B) posterior": res.posterior,
            "Δ posterior−prior": delta,
        })
        indices.append(col)

    hm_df = pd.DataFrame(rows, index=indices)

    n_rows = len(hm_df)
    fig, ax = new_figure(figsize=(8, max(4, n_rows * 0.5 + 2)))

    # Colormap divergente centrado en 0 para la columna Δ
    sns.heatmap(
        hm_df,
        annot=True, fmt=".4f",
        cmap=_diverging_cmap(),
        center=0,
        linewidths=0.5, linecolor=PALETTE.bg,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Valor / Δ"},
        annot_kws={"fontsize": 8, "color": PALETTE.text},
    )

    # Estilo del colorbar
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color=PALETTE.subtext)
            cbar.ax.yaxis.label.set_color(PALETTE.subtext)

    theme.apply_axes(ax, title=f"Prior vs Posterior — {multi.target}")
    fig.tight_layout(pad=2.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helper privado
# ═════════════════════════════════════════════════════════════════════════════

def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "bayes_div",
        [PALETTE.teal, PALETTE.ax_bg, PALETTE.red],
    )