"""
temporal.py
===========
Responsabilidad única: graficar la evolución temporal de una o varias
columnas numéricas usando la columna 'Date' generada por TimestampBuilder.

Entrada esperada
────────────────
    · pd.DataFrame con columna 'Date' (datetime64) y columnas numéricas
    · nombre(s) de columna(s) a trazar

Salida
──────
    · matplotlib.figure.Figure lista para Tkinter o savefig()

No contiene lógica de probabilidad ni de preprocesamiento.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from .style import PALETTE, theme, new_figure, new_figure_grid


# ═════════════════════════════════════════════════════════════════════════════
# Gráfica de serie temporal — una columna con anotaciones
# ═════════════════════════════════════════════════════════════════════════════

def plot_temporal(
    df:         pd.DataFrame,
    column:     str,
    date_col:   str   = "Date",
    threshold:  float | None = None,
    target:     str   | None = None,
    window:     int   = 7,
) -> matplotlib.figure.Figure:
    """
    Traza la evolución diaria de una variable numérica.

    Parámetros
    ──────────
    df         : DataFrame preprocesado con columna Date
    column     : columna numérica a graficar en el eje Y
    date_col   : nombre de la columna datetime (por defecto 'Date')
    threshold  : si se indica, dibuja una línea horizontal de umbral
    target     : si se indica, marca en rojo los días donde target == 1
    window     : tamaño de ventana para la media móvil (0 = desactivar)

    Retorna
    ───────
    matplotlib.figure.Figure
    """
    _validate(df, date_col, column)
    data = df[[date_col, column]].dropna().sort_values(date_col)

    fig, ax = new_figure(figsize=(10, 4))

    # — Serie principal ──────────────────────────────────────────────────
    ax.plot(data[date_col], data[column],
            color=PALETTE.blue, linewidth=1.4, alpha=0.7,
            label=column, zorder=2)

    # — Media móvil ──────────────────────────────────────────────────────
    if window > 1 and len(data) >= window:
        rolling = data[column].rolling(window, center=True).mean()
        ax.plot(data[date_col], rolling,
                color=PALETTE.lavender, linewidth=2.0,
                label=f"Media móvil ({window}d)", zorder=3)

    # — Línea de umbral ──────────────────────────────────────────────────
    if threshold is not None:
        ax.axhline(threshold, color=PALETTE.yellow,
                   linestyle="--", linewidth=1.2,
                   label=f"Umbral = {threshold:.2f}", zorder=1)
        # Sombrear área por encima del umbral
        ax.fill_between(data[date_col], threshold, data[column],
                        where=(data[column] > threshold),
                        color=PALETTE.red, alpha=0.12, zorder=1)

    # — Marcadores de evento (target == 1) ───────────────────────────────
    if target and target in df.columns:
        event_mask = df[target] == 1
        event_data = df.loc[event_mask, [date_col, column]].dropna()
        if not event_data.empty:
            ax.scatter(event_data[date_col], event_data[column],
                       color=PALETTE.red, s=40, zorder=5,
                       label=f"{target} = 1", alpha=0.85)

    # — Formato del eje X (fechas) ───────────────────────────────────────
    _format_date_axis(ax, data[date_col])

    ax.set_xlabel("Fecha")
    ax.set_ylabel(column)
    ax.legend(loc="upper left")
    theme.apply_axes(ax, title=f"Serie temporal — {column}")
    fig.tight_layout(pad=2.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Gráfica multi-columna (panel temporal comparativo)
# ═════════════════════════════════════════════════════════════════════════════

def plot_temporal_grid(
    df:        pd.DataFrame,
    columns:   list[str],
    date_col:  str  = "Date",
    target:    str | None = None,
) -> matplotlib.figure.Figure:
    """
    Grid de series temporales, una por columna.
    Comparten el eje X (fecha) para facilitar comparación visual.

    Parámetros
    ──────────
    df      : DataFrame preprocesado
    columns : columnas numéricas a mostrar
    date_col: columna datetime
    target  : si se indica, sombrea los días donde target == 1

    Retorna
    ───────
    matplotlib.figure.Figure
    """
    cols = [c for c in columns if c in df.columns and c != date_col]
    if not cols:
        raise ValueError("Ninguna de las columnas indicadas existe en el DataFrame.")

    n     = len(cols)
    fig, axes = new_figure_grid(n, 1, figsize=(10, 3 * n))

    data_sorted = df.sort_values(date_col)

    for idx, col in enumerate(cols):
        ax     = axes[idx]
        series = data_sorted[[date_col, col]].dropna()

        ax.plot(series[date_col], series[col],
                color=PALETTE.sequence[idx % len(PALETTE.sequence)],
                linewidth=1.4, alpha=0.85)

        # Sombrear eventos
        if target and target in df.columns:
            _shade_events(ax, data_sorted, date_col, col, target)

        _format_date_axis(ax, series[date_col])
        ax.set_ylabel(col)
        theme.apply_axes(ax, title=col)

        # Solo el último eje muestra el label del eje X
        if idx < n - 1:
            ax.set_xlabel("")

    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Evolución temporal de variables",
                 color=PALETTE.text, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helpers privados
# ═════════════════════════════════════════════════════════════════════════════

def _format_date_axis(ax: plt.Axes, dates: pd.Series) -> None:
    """Formatea el eje X con fechas legibles según el rango temporal."""
    date_range = (dates.max() - dates.min()).days if len(dates) > 1 else 0

    if date_range <= 60:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    elif date_range <= 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    ax.figure.autofmt_xdate(rotation=30, ha="right")


def _shade_events(
    ax:       plt.Axes,
    df:       pd.DataFrame,
    date_col: str,
    col:      str,
    target:   str,
) -> None:
    """Sombrea verticalmente los días donde target == 1."""
    event_dates = df.loc[df[target] == 1, date_col].dropna()
    y_min, y_max = df[col].min(), df[col].max()
    for date in event_dates:
        ax.axvspan(date, date + pd.Timedelta(days=1),
                   color=PALETTE.red, alpha=0.10, zorder=0)


def _validate(df: pd.DataFrame, date_col: str, column: str) -> None:
    if date_col not in df.columns:
        raise KeyError(
            f"La columna de fechas '{date_col}' no existe. "
            "Asegúrate de ejecutar DataPreprocessor antes de graficar."
        )
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError(f"'{date_col}' no es de tipo datetime64.")