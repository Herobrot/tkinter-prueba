"""
src/ui/panels/charts.py
=======================
Responsabilidad única: panel principal de gráficas con pestañas (ttk.Notebook).

Cada pestaña contiene un ChartCanvas independiente.
ChartsPanel recibe figuras ya generadas por el controller y las
distribuye en la pestaña correspondiente.

Pestañas
────────
    0 · Histogramas          → distribución de variables numéricas
    1 · Serie Temporal       → evolución diaria de la evidencia seleccionada
    2 · Posterior (simple)   → P(A) vs P(A|B) — una evidencia
    3 · Posterior (multi)    → P(A|Bᵢ) para todas las evidencias
    4 · Descomposición Bayes → prior / likelihood / marginal / posterior
    5 · Conf. + Métricas     → panel combinado de Naive Bayes
    6 · Comparación          → barras agrupadas prior vs posterior
    7 · Lift                 → lift bayesiano por evidencia
    8 · Heatmap              → mapa de calor prior vs posterior
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import matplotlib.figure

from src.ui.theme   import T
from src.ui.widgets import StyledFrame
from src.ui.canvas  import ChartCanvas


# Definición declarativa de las pestañas
_TABS: list[tuple[str, str]] = [
    ("hist",        "📊  Histogramas"),
    ("temporal",    "📅  Serie Temporal"),
    ("posterior",   "🎯  Posterior (simple)"),
    ("post_multi",  "🎯  Posterior (multi)"),
    ("breakdown",   "🔍  Desglose Bayes"),
    ("confusion",   "⚖️  Clasificador"),
    ("comparison",  "📈  Comparación"),
    ("lift",        "🚀  Lift"),
    ("heatmap",     "🗺️  Heatmap"),
]


class ChartsPanel(StyledFrame):
    """
    Panel con ttk.Notebook que contiene un ChartCanvas por pestaña.

    Uso
    ───
        charts = ChartsPanel(parent)
        charts.pack(fill=tk.BOTH, expand=True)

        # Después de calcular:
        charts.update("hist",       fig_histogramas)
        charts.update("temporal",   fig_temporal)
        charts.update("posterior",  fig_posterior_single)
        ...
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=T.bg, **kwargs)

        self._canvases: dict[str, ChartCanvas] = {}
        self._build()

    # ── Construcción ──────────────────────────────────────────────────────

    def _build(self) -> None:
        # Estilo del Notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TNotebook",
            background=T.panel,
            borderwidth=0,
            tabmargins=[2, 2, 2, 0],
        )
        style.configure(
            "Dark.TNotebook.Tab",
            background=T.surface,
            foreground=T.subtext,
            font=T.font_sm,
            padding=[T.pad_sm, T.pad_xs],
            borderwidth=0,
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", T.bg)],
            foreground=[("selected", T.blue)],
        )

        self._notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True)

        for key, label in _TABS:
            frame  = StyledFrame(self._notebook, bg=T.bg)
            canvas = ChartCanvas(frame, show_toolbar=True)
            canvas.pack(fill=tk.BOTH, expand=True)
            self._notebook.add(frame, text=label)
            self._canvases[key] = canvas

    # ── API pública ───────────────────────────────────────────────────────

    def update(self, tab_key: str, fig: matplotlib.figure.Figure) -> None:
        """
        Renderiza una figura en la pestaña identificada por tab_key.
        tab_key debe ser uno de los keys definidos en _TABS.
        """
        if tab_key not in self._canvases:
            raise KeyError(f"Pestaña '{tab_key}' no existe. "
                           f"Claves válidas: {list(self._canvases)}")
        self._canvases[tab_key].update_figure(fig)

    def clear_all(self) -> None:
        """Vuelve todos los canvases al estado placeholder."""
        for canvas in self._canvases.values():
            canvas.clear()

    def select_tab(self, tab_key: str) -> None:
        """Activa programáticamente una pestaña por su key."""
        keys = [k for k, _ in _TABS]
        if tab_key in keys:
            idx = keys.index(tab_key)
            self._notebook.select(idx)