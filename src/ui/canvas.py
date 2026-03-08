"""
src/ui/canvas.py
================
Responsabilidad única: embeber una matplotlib.figure.Figure dentro de
un Frame de Tkinter usando FigureCanvasTkAgg.

Expone ChartCanvas, que es un Frame especializado que:
    · Renderiza cualquier Figure pasada por update_figure()
    · Libera la figura anterior antes de renderizar la nueva
    · Muestra un placeholder mientras no hay figura cargada
    · No sabe nada de probabilidad ni de datos

Uso
───
    canvas = ChartCanvas(parent_frame)
    canvas.pack(fill=tk.BOTH, expand=True)

    fig = plot_histograms(df, cols)
    canvas.update_figure(fig)
"""

from __future__ import annotations

import tkinter as tk
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .theme import T
from .widgets import StyledFrame


class ChartCanvas(StyledFrame):
    """
    Frame Tkinter que contiene un canvas de matplotlib.

    Parámetros
    ──────────
    parent          : widget padre Tkinter
    show_toolbar    : si True, muestra la barra de navegación de matplotlib
                      (zoom, pan, save) debajo del gráfico
    placeholder_text: texto mostrado cuando no hay figura cargada
    """

    def __init__(
        self,
        parent,
        show_toolbar:     bool = True,
        placeholder_text: str  = "Selecciona una variable objetivo y pulsa  ▶  Calcular",
        **kwargs,
    ):
        super().__init__(parent, bg=T.bg, **kwargs)

        self._canvas:  FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._show_toolbar = show_toolbar

        # Placeholder visible antes de la primera figura
        self._placeholder = tk.Label(
            self,
            text=placeholder_text,
            bg=T.bg,
            fg=T.disabled,
            font=("Segoe UI", 11),
        )
        self._placeholder.pack(expand=True)

    # ── API pública ───────────────────────────────────────────────────────

    def update_figure(self, fig: matplotlib.figure.Figure) -> None:
        """
        Reemplaza el canvas con la nueva Figure.
        Destruye el canvas anterior y libera memoria de matplotlib.
        """
        self._clear()

        self._placeholder.pack_forget()

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()

        widget = self._canvas.get_tk_widget()
        widget.configure(bg=T.bg)
        widget.pack(fill=tk.BOTH, expand=True)

        if self._show_toolbar:
            toolbar_frame = StyledFrame(self, bg=T.panel)
            toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
            self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
            self._toolbar.configure(background=T.panel)
            self._toolbar.update()
            # Estilizar los botones de la toolbar
            for child in self._toolbar.winfo_children():
                try:
                    child.configure(background=T.panel, foreground=T.subtext,
                                    relief=tk.FLAT)
                except tk.TclError:
                    pass

        plt.close(fig)

    def clear(self) -> None:
        """Vuelve al estado placeholder."""
        self._clear()
        self._placeholder.pack(expand=True)

    # ── Helpers privados ──────────────────────────────────────────────────

    def _clear(self) -> None:
        """Destruye el canvas y toolbar actuales si existen."""
        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None