"""
src/ui/app.py
=============
Responsabilidad única: ensamblar paneles y controller en la ventana
principal de Tkinter, y coordinar el nuevo flujo de trabajo:

    1. Usuario pulsa "📂 Abrir CSV"
    2. app.py llama a controller.load_data(filepath)  en hilo separado
    3. Al terminar la carga, se refresca la sidebar y se dispara compute()
    4. compute() corre en otro hilo y rellena todas las gráficas
    5. El usuario puede cambiar target/evidencia → se dispara compute() de nuevo
    6. "▶ Recalcular" permite forzar el análisis en cualquier momento

No contiene lógica de probabilidad ni de graficación.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from src.ui.theme            import T
from src.ui.state            import AppState
from src.ui.controller       import BayesController
from src.ui.widgets          import StyledFrame, HeaderBar
from src.ui.panels.sidebar   import SidebarPanel
from src.ui.panels.charts    import ChartsPanel
from src.ui.panels.statusbar import StatusBar


class App(tk.Tk):
    """
    Ventana principal de Weather Bayes.

    Layout
    ──────
    ┌─────────────────────────────────────────────────┐
    │  HeaderBar                                      │
    ├──────────────┬──────────────────────────────────┤
    │              │                                  │
    │  SidebarPanel│  ChartsPanel (Notebook 9 tabs)   │
    │              │                                  │
    ├──────────────┴──────────────────────────────────┤
    │  StatusBar                                      │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()

        self.title("Weather Bayes — Análisis Bayesiano")
        self.configure(bg=T.bg)
        self.geometry("1400x820")
        self.minsize(900, 600)

        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        self._state      = AppState()
        self._controller = BayesController(self._state)

        # Bloqueo para evitar computes solapados en hilos
        self._computing = False

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self._header = HeaderBar(self, title="🌦  Weather Bayes")
        self._header.pack(fill=tk.X, side=tk.TOP)

        self._statusbar = StatusBar(self)
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        body = StyledFrame(self, bg=T.bg)
        body.pack(fill=tk.BOTH, expand=True)

        self._sidebar = SidebarPanel(
            body,
            state            = self._state,
            on_file_selected = self._on_file_selected,
            on_compute       = self._on_compute,
        )
        self._sidebar.pack(fill=tk.Y, side=tk.LEFT)

        tk.Frame(body, bg=T.surface2, width=1).pack(fill=tk.Y, side=tk.LEFT)

        self._charts = ChartsPanel(body)
        self._charts.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self._statusbar.set_message(
            "Listo — abre un archivo CSV con  📂  Abrir CSV", "info"
        )

    # ── Flujo de carga de archivo ─────────────────────────────────────────

    def _on_file_selected(self, filepath: str) -> None:
        """
        Callback desde SidebarPanel cuando el usuario eligió un CSV.
        Carga los datos en un hilo y, al terminar, dispara el análisis.
        """
        filename = Path(filepath).name
        self._statusbar.set_message(f"Cargando  {filename}…", "info")
        self._statusbar.set_loading(True)
        self._sidebar.clear_results()
        self._charts.clear_all()

        def _worker():
            try:
                self._controller.load_data(filepath)
                self.after(0, lambda: self._on_data_loaded(filename))
            except Exception as exc:
                self.after(0, lambda: self._on_load_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_data_loaded(self, filename: str) -> None:
        """Ejecutado en el hilo principal tras cargar datos correctamente."""
        s = self._state
        self._statusbar.set_loading(False)
        self._statusbar.set_message(s.status_msg, "success")

        self._sidebar.set_file_label(filename)
        self._header.set_subtitle(
            f"{len(s.df)} registros  ·  "
            f"{s.df['Date'].min().date()} → {s.df['Date'].max().date()}"
        )

        # Poblar los combos (incluye filtrado de compatibilidad)
        self._sidebar.refresh_combos()
        self._statusbar.set_message(
            "Datos cargados — configura las variables y pulsa  ▶  Recalcular", "success"
        )

    def _on_load_error(self, msg: str) -> None:
        self._statusbar.set_loading(False)
        self._statusbar.set_message(f"Error al cargar: {msg}", "error")
        messagebox.showerror(
            "Error de carga",
            f"No se pudo leer el archivo:\n\n{msg}"
        )

    # ── Flujo de cálculo ──────────────────────────────────────────────────

    def _on_compute(self) -> None:
        """
        Ejecuta el análisis bayesiano completo en un hilo separado.
        Si ya hay un cómputo en curso lo ignora (no encola).
        """
        if not self._state.loaded:
            return
        if self._computing:
            return

        self._computing = True
        self._sidebar.set_controls_enabled(False)
        self._statusbar.set_message("Calculando…", "info")
        self._statusbar.set_loading(True)
        self._sidebar.clear_results()

        def _worker():
            try:
                self._controller.compute()
                self.after(0, self._on_compute_done)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: self._on_compute_error(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_compute_done(self) -> None:
        """Ejecutado en el hilo principal tras completar el cómputo."""
        self._computing = False
        ctrl = self._controller

        updates = [
            ("hist",       ctrl.fig_histograms),
            ("temporal",   ctrl.fig_temporal),
            ("posterior",  ctrl.fig_posterior_single),
            ("post_multi", ctrl.fig_posterior_multi),
            ("breakdown",  ctrl.fig_posterior_breakdown),
            ("confusion",  ctrl.fig_confusion_panel),
            ("comparison", ctrl.fig_prior_vs_posterior),
            ("lift",       ctrl.fig_lift),
            ("heatmap",    ctrl.fig_heatmap),
        ]

        for tab_key, fig_fn in updates:
            try:
                self._charts.update(tab_key, fig_fn())
            except Exception as exc:
                print(f"[warn] figura '{tab_key}' falló: {exc}")

        self._sidebar.update_results(ctrl.build_results_lines())
        self._charts.select_tab("posterior")

        self._sidebar.set_controls_enabled(True)
        self._statusbar.set_loading(False)
        self._statusbar.set_message(self._state.status_msg, "success")

    def _on_compute_error(self, msg: str) -> None:
        self._computing = False
        self._sidebar.set_controls_enabled(True)
        self._statusbar.set_loading(False)
        self._statusbar.set_message(f"Error en el cálculo: {msg}", "error")
        messagebox.showerror(
            "Error de cálculo",
            f"Ocurrió un error durante el análisis:\n\n{msg}"
        )