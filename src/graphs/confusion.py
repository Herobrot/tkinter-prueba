"""
confusion.py
============
Responsabilidad única: visualizar la matriz de confusión y las métricas
derivadas del clasificador Naive Bayes.

Consume exclusivamente los Value Objects ConfusionMatrix y ClassifierMetrics
del módulo src/maths/naive_bayes.py; no realiza cálculos propios.

Gráficas disponibles
────────────────────
    plot_confusion_matrix  → heatmap 2×2 con anotaciones y métricas
    plot_metrics_bar       → barras horizontales de accuracy/sens/espec/prec
    plot_confusion_panel   → panel combinado (heatmap + barras)
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .style import PALETTE, theme, new_figure, new_figure_grid
from src.maths.naive_bayes import ConfusionMatrix, ClassifierMetrics, NaiveBayesResult


# ═════════════════════════════════════════════════════════════════════════════
# Heatmap de la matriz de confusión
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm:     ConfusionMatrix,
    target: str = "Evento",
) -> matplotlib.figure.Figure:
    """
    Heatmap 2×2 de la matriz de confusión con anotaciones absolutas
    y porcentaje relativo al total de muestras.

    Parámetros
    ──────────
    cm     : ConfusionMatrix de ClassifierMetrics.confusion
    target : nombre del evento objetivo (para etiquetas de ejes)
    """
    fig, ax = new_figure(figsize=(5.5, 4.5))

    matrix_df = cm.as_dataframe()
    n_total   = cm.tp + cm.fp + cm.tn + cm.fn

    # Anotaciones: valor absoluto + porcentaje
    annot = np.array([
        [f"{cm.tp}\n({cm.tp/n_total:.1%})" if n_total else "0",
         f"{cm.fp}\n({cm.fp/n_total:.1%})" if n_total else "0"],
        [f"{cm.fn}\n({cm.fn/n_total:.1%})" if n_total else "0",
         f"{cm.tn}\n({cm.tn/n_total:.1%})" if n_total else "0"],
    ])

    sns.heatmap(
        matrix_df,
        annot=annot, fmt="",
        cmap=_build_confusion_cmap(),
        linewidths=1.5, linecolor=PALETTE.bg,
        ax=ax,
        cbar=False,
        annot_kws={"fontsize": 12, "fontweight": "bold", "color": PALETTE.text},
    )

    # Colorear celdas TP/TN en verde y FP/FN en rojo
    _color_confusion_cells(ax, cm)

    ax.set_xlabel(f"Predicción  ({target})", labelpad=8)
    ax.set_ylabel(f"Valor real  ({target})", labelpad=8)
    theme.apply_axes(ax, title="Matriz de Confusión")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Barras de métricas
# ═════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(metrics: ClassifierMetrics) -> matplotlib.figure.Figure:
    """
    Barras horizontales con accuracy, sensibilidad, especificidad y precisión.
    Cada barra incluye una línea de referencia en 0.5 y anotación numérica.

    Parámetro
    ─────────
    metrics : ClassifierMetrics devuelto por NaiveBayesClassifier.run()
    """
    fig, ax = new_figure(figsize=(7, 4))

    series = metrics.as_series()
    colors = [
        PALETTE.green   if v >= 0.75 else
        PALETTE.yellow  if v >= 0.50 else
        PALETTE.red
        for v in series.values
    ]

    bars = ax.barh(series.index, series.values,
                   color=colors, alpha=0.85, height=0.5, zorder=3)

    # Línea de referencia en 0.5
    ax.axvline(0.5, color=PALETTE.subtext, linestyle=":",
               linewidth=1.2, alpha=0.7, label="Ref. 0.5")

    # Anotaciones
    for bar, val in zip(bars, series.values):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center",
                color=PALETTE.text, fontsize=9, fontweight="bold")

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Valor")
    ax.legend(loc="lower right")
    theme.apply_axes(ax, title="Métricas del Clasificador Naive Bayes")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Panel combinado (heatmap + métricas + distribución de probabilidades)
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion_panel(nb_result: NaiveBayesResult) -> matplotlib.figure.Figure:
    """
    Panel de 3 gráficas en una figura:
        izq  : heatmap de la matriz de confusión
        centro: barras de métricas
        der  : histograma de P(clase=1|x) separado por clase real

    Parámetro
    ─────────
    nb_result : NaiveBayesResult devuelto por NaiveBayesClassifier.run()
    """
    fig, axes = new_figure_grid(1, 3, figsize=(14, 5))
    ax_cm, ax_metrics, ax_prob = axes

    metrics = nb_result.metrics
    cm      = metrics.confusion
    n_total = cm.tp + cm.fp + cm.tn + cm.fn

    # — Heatmap ──────────────────────────────────────────────────────────
    annot = np.array([
        [f"{cm.tp}\n({cm.tp/n_total:.1%})" if n_total else "0",
         f"{cm.fp}\n({cm.fp/n_total:.1%})" if n_total else "0"],
        [f"{cm.fn}\n({cm.fn/n_total:.1%})" if n_total else "0",
         f"{cm.tn}\n({cm.tn/n_total:.1%})" if n_total else "0"],
    ])
    sns.heatmap(
        cm.as_dataframe(), annot=annot, fmt="",
        cmap=_build_confusion_cmap(),
        linewidths=1.5, linecolor=PALETTE.bg,
        ax=ax_cm, cbar=False,
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": PALETTE.text},
    )
    _color_confusion_cells(ax_cm, cm)
    theme.apply_axes(ax_cm, title="Matriz de Confusión")

    # — Métricas ─────────────────────────────────────────────────────────
    series = metrics.as_series()
    colors = [
        PALETTE.green  if v >= 0.75 else
        PALETTE.yellow if v >= 0.50 else
        PALETTE.red
        for v in series.values
    ]
    bars = ax_metrics.barh(series.index, series.values,
                            color=colors, alpha=0.85, height=0.5, zorder=3)
    ax_metrics.axvline(0.5, color=PALETTE.subtext, linestyle=":",
                       linewidth=1.2, alpha=0.7)
    for bar, val in zip(bars, series.values):
        ax_metrics.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=PALETTE.text, fontsize=8,
        )
    ax_metrics.set_xlim(0, 1.18)
    theme.apply_axes(ax_metrics, title="Métricas")

    # — Distribución de P(clase=1|x) ─────────────────────────────────────
    pred_df = nb_result.prediction_dataframe()
    for cls, color, label in (
        (0, PALETTE.blue, "Real: 0"),
        (1, PALETTE.red,  "Real: 1"),
    ):
        subset = pred_df[pred_df["y_true"] == cls]["p_positive"].dropna()
        if len(subset) > 0:
            ax_prob.hist(subset, bins=15, color=color,
                         alpha=0.60, label=label, edgecolor=PALETTE.ax_bg)
    ax_prob.axvline(0.5, color=PALETTE.yellow, linestyle="--",
                    linewidth=1.3, label="Umbral 0.5")
    ax_prob.set_xlabel("P(evento = 1 | x₁…xₙ)")
    ax_prob.set_ylabel("Frecuencia")
    ax_prob.legend()
    theme.apply_axes(ax_prob, title="Distribución de Probabilidad Posterior")

    fig.suptitle(
        f"Evaluación Naive Bayes — {nb_result.target}",
        color=PALETTE.text, fontsize=12, fontweight="bold",
    )
    fig.tight_layout(pad=2.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helpers privados
# ═════════════════════════════════════════════════════════════════════════════

def _build_confusion_cmap():
    """Colormap personalizado para el heatmap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "confusion",
        [PALETTE.ax_bg, PALETTE.surface],
    )


def _color_confusion_cells(ax, cm: ConfusionMatrix) -> None:
    """
    Pinta manualmente el fondo de las celdas:
        TP, TN → verde (correcto)
        FP, FN → rojo  (error)
    """
    cell_colors = [
        (PALETTE.green, 0.25),   # TP  (0,0)
        (PALETTE.red,   0.20),   # FP  (0,1)
        (PALETTE.red,   0.20),   # FN  (1,0)
        (PALETTE.green, 0.25),   # TN  (1,1)
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]    
    for (row, col), (color, alpha) in zip(positions, cell_colors):
        ax.add_patch(
            plt.Rectangle((col, row), 1, 1,
                           fill=True, color=color,
                           alpha=alpha, zorder=0)
        )