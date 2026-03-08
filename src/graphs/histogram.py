"""
histogram.py
============
Responsabilidad única: generar histogramas de las columnas numéricas
del DataFrame preprocesado.

Entrada esperada
────────────────
    · pd.DataFrame con las columnas numéricas del WeatherSchema
    · lista de columnas a graficar (subset de WeatherSchema.numeric_cols)

Salida
──────
    · matplotlib.figure.Figure lista para embeber en Tkinter con
      FigureCanvasTkAgg, o para guardar con fig.savefig().

No contiene lógica de negocio ni de probabilidad.
"""

from __future__ import annotations

import math

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import PALETTE, theme, new_figure, new_figure_grid


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de layout
# ─────────────────────────────────────────────────────────────────────────────
_MAX_COLS_PER_ROW = 4
_DEFAULT_BINS     = 20


# ═════════════════════════════════════════════════════════════════════════════
# Gráfica principal
# ═════════════════════════════════════════════════════════════════════════════

def plot_histograms(
    df:       pd.DataFrame,
    columns:  list[str],
    bins:     int   = _DEFAULT_BINS,
    kde:      bool  = True,
    target:   str | None = None,
) -> matplotlib.figure.Figure:
    """
    Genera un grid de histogramas, uno por columna numérica.

    Parámetros
    ──────────
    df      : DataFrame preprocesado
    columns : columnas numéricas a graficar
    bins    : número de bins para cada histograma
    kde     : superponer curva de densidad KDE
    target  : si se indica, colorea las barras según la clase 0/1 del target
              (útil para ver separación entre clases)

    Retorna
    ───────
    matplotlib.figure.Figure — embebible directamente en Tkinter
    """
    cols  = [c for c in columns if c in df.columns]
    n     = len(cols)
    if n == 0:
        raise ValueError("Ninguna de las columnas indicadas existe en el DataFrame.")

    ncols = min(n, _MAX_COLS_PER_ROW)
    nrows = math.ceil(n / ncols)
    fig, axes = new_figure_grid(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))

    for idx, col in enumerate(cols):
        ax = axes[idx]
        series = df[col].dropna()

        if target and target in df.columns:
            # Histograma separado por clase
            for cls, color, label in (
                (0, PALETTE.blue,  f"{target}=0"),
                (1, PALETTE.red,   f"{target}=1"),
            ):
                mask = df[target] == cls
                data = df.loc[mask, col].dropna()
                if len(data) == 0:
                    continue
                ax.hist(data, bins=bins, color=color, alpha=0.55,
                        label=label, edgecolor=PALETTE.ax_bg, linewidth=0.4)
                if kde and len(data) > 2:
                    _overlay_kde(ax, data, color)
            ax.legend(fontsize=7)
        else:
            ax.hist(series, bins=bins, color=PALETTE.blue,
                    alpha=0.80, edgecolor=PALETTE.ax_bg, linewidth=0.4)
            if kde and len(series) > 2:
                _overlay_kde(ax, series, PALETTE.blue)

        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        theme.apply_axes(ax, title=col)

    # Ocultar ejes sobrantes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribución de variables numéricas",
                 color=PALETTE.text, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Histograma de una sola variable (para panel detallado en Tkinter)
# ═════════════════════════════════════════════════════════════════════════════

def plot_single_histogram(
    df:     pd.DataFrame,
    column: str,
    bins:   int  = _DEFAULT_BINS,
    kde:    bool = True,
) -> matplotlib.figure.Figure:
    """
    Histograma de una única columna numérica con estadísticos superpuestos.

    Marca μ (media) y μ ± σ como líneas verticales punteadas.
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

    series = df[column].dropna()
    fig, ax = new_figure(figsize=(7, 4))

    ax.hist(series, bins=bins, color=PALETTE.blue,
            alpha=0.80, edgecolor=PALETTE.ax_bg, linewidth=0.4,
            label="Frecuencia")

    if kde and len(series) > 2:
        _overlay_kde(ax, series, PALETTE.blue)

    # Líneas de μ y σ
    mu, sigma = series.mean(), series.std()
    ax.axvline(mu,         color=PALETTE.red,    linestyle="--",
               linewidth=1.4, label=f"μ = {mu:.2f}")
    ax.axvline(mu - sigma, color=PALETTE.yellow, linestyle=":",
               linewidth=1.2, label=f"μ−σ = {mu - sigma:.2f}")
    ax.axvline(mu + sigma, color=PALETTE.yellow, linestyle=":",
               linewidth=1.2, label=f"μ+σ = {mu + sigma:.2f}")

    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")
    ax.legend()
    theme.apply_axes(ax, title=f"Histograma — {column}")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helper privado
# ═════════════════════════════════════════════════════════════════════════════

def _overlay_kde(
    ax:     plt.Axes,
    series: pd.Series,
    color:  str,
) -> None:
    """
    Superpone una curva KDE gaussiana sobre un histograma existente.
    Escala la curva al eje Y de frecuencias absolutas.
    """
    from scipy.stats import gaussian_kde   # import diferido — solo si se usa KDE

    data   = series.to_numpy(dtype=float)
    kde_fn = gaussian_kde(data)
    x_min, x_max = data.min(), data.max()
    x_range = np.linspace(x_min, x_max, 300)

    # Escalar KDE a frecuencia absoluta
    bin_width = (x_max - x_min) / _DEFAULT_BINS
    scale     = len(data) * bin_width
    ax.plot(x_range, kde_fn(x_range) * scale,
            color=color, linewidth=1.8, alpha=0.9)