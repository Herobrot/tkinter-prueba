"""
style.py
========
Responsabilidad única: definir y aplicar el tema visual compartido por
todas las gráficas del paquete src/graphs.

Ningún otro módulo define colores, fuentes ni tamaños directamente;
todos importan desde aquí para garantizar coherencia visual en Tkinter.

Uso
───
    from src.graphs.style import theme, new_figure

    fig, ax = new_figure(figsize=(6, 4))
    theme.apply_axes(ax)
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns


# ═════════════════════════════════════════════════════════════════════════════
# Paleta de colores (Catppuccin Mocha — legible en fondos oscuros y claros)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Palette:
    # Fondos
    bg:       str = "#1e1e2e"   # fondo de figura
    ax_bg:    str = "#181825"   # fondo de ejes
    surface:  str = "#313244"   # superficies secundarias

    # Texto y líneas
    text:     str = "#cdd6f4"
    subtext:  str = "#a6adc8"
    grid:     str = "#45475a"

    # Acentos principales
    blue:     str = "#89b4fa"   # P(A), prior, barras principales
    red:      str = "#f38ba8"   # P(Fallo), posterior, alarma
    green:    str = "#a6e3a1"   # verdadero positivo, accuracy
    yellow:   str = "#f9e2af"   # verdadero negativo
    lavender: str = "#b4befe"   # acento secundario
    teal:     str = "#94e2d5"   # tercera variable

    # Secuencia para múltiples series
    @property
    def sequence(self) -> list[str]:
        return [self.blue, self.red, self.green, self.yellow,
                self.lavender, self.teal]


PALETTE = Palette()


# ═════════════════════════════════════════════════════════════════════════════
# Tema global
# ═════════════════════════════════════════════════════════════════════════════

class GraphTheme:
    """
    Encapsula la configuración de matplotlib/seaborn.
    Se aplica una vez al importar el módulo (via apply_global())
    y puede re-aplicarse a ejes individuales con apply_axes().
    """

    FONT_FAMILY = "Segoe UI"
    FONT_SIZE   = 9
    TITLE_SIZE  = 10
    TICK_SIZE   = 8

    def __init__(self, palette: Palette = PALETTE) -> None:
        self.p = palette

    def apply_global(self) -> None:
        """
        Aplica rcParams globales de matplotlib y el tema de seaborn.
        Llamar una sola vez al importar el paquete.
        """
        mpl.rcParams.update({
            # Figura
            "figure.facecolor":  self.p.bg,
            "figure.edgecolor":  self.p.bg,

            # Ejes
            "axes.facecolor":    self.p.ax_bg,
            "axes.edgecolor":    self.p.subtext,
            "axes.labelcolor":   self.p.subtext,
            "axes.titlecolor":   self.p.text,
            "axes.titlesize":    self.TITLE_SIZE,
            "axes.labelsize":    self.FONT_SIZE,
            "axes.grid":         True,
            "axes.spines.top":   False,
            "axes.spines.right": False,

            # Grid
            "grid.color":        self.p.grid,
            "grid.linewidth":    0.5,
            "grid.alpha":        0.6,

            # Ticks
            "xtick.color":       self.p.subtext,
            "ytick.color":       self.p.subtext,
            "xtick.labelsize":   self.TICK_SIZE,
            "ytick.labelsize":   self.TICK_SIZE,

            # Leyenda
            "legend.facecolor":  self.p.surface,
            "legend.edgecolor":  self.p.grid,
            "legend.labelcolor": self.p.text,
            "legend.fontsize":   self.FONT_SIZE,

            # Texto global
            "text.color":        self.p.text,
            "font.family":       "sans-serif",
            "font.size":         self.FONT_SIZE,

            # Líneas
            "lines.linewidth":   1.6,
        })

        sns.set_theme(
            style="darkgrid",
            rc={
                "axes.facecolor": self.p.ax_bg,
                "figure.facecolor": self.p.bg,
                "grid.color": self.p.grid,
            },
        )

    def apply_axes(self, ax: mpl.axes.Axes, title: str = "") -> None:
        """
        Re-aplica estilos a un eje específico.
        Útil cuando matplotlib regenera el eje (ej. colorbar en heatmap).
        """
        ax.set_facecolor(self.p.ax_bg)
        if title:
            ax.set_title(title, color=self.p.text,
                         fontsize=self.TITLE_SIZE, pad=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(self.p.subtext)
        ax.tick_params(colors=self.p.subtext, labelsize=self.TICK_SIZE)
        ax.xaxis.label.set_color(self.p.subtext)
        ax.yaxis.label.set_color(self.p.subtext)


# Instancia global — todos los módulos la importan
theme = GraphTheme(PALETTE)
theme.apply_global()


# ═════════════════════════════════════════════════════════════════════════════
# Fábrica de Figure
# ═════════════════════════════════════════════════════════════════════════════

def new_figure(
    figsize: tuple[float, float] = (8, 5),
    tight: bool = True,
) -> tuple[matplotlib.figure.Figure, mpl.axes.Axes]:
    """
    Crea una Figure + un único Axes con el tema aplicado.
    Para figuras con múltiples subplots usar new_figure_grid().
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE.bg)
    ax.set_facecolor(PALETTE.ax_bg)
    if tight:
        fig.tight_layout(pad=2.0)
    return fig, ax


def new_figure_grid(
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] | None = None,
    tight: bool = True,
) -> tuple[matplotlib.figure.Figure, list[mpl.axes.Axes]]:
    """
    Crea una Figure con una cuadrícula de Axes.
    Devuelve (fig, lista_plana_de_axes).
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=figsize,
                                  facecolor=PALETTE.bg)
    axes: list[mpl.axes.Axes] = (
        axes_grid.flatten().tolist()
        if hasattr(axes_grid, "flatten")
        else [axes_grid]
    )
    for ax in axes:
        ax.set_facecolor(PALETTE.ax_bg)

    if tight:
        fig.tight_layout(pad=2.5)
    return fig, axes