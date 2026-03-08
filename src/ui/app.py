"""
src/ui/app.py
=============
Responsabilidad única: ensamblar todos los paneles y el controller en
la ventana principal de Tkinter.

App es el único lugar donde:
    · Se instancian AppState y BayesController
    · Se conectan los callbacks de los paneles al controller
    · Se orquestan las llamadas en el orden correcto

No contiene lógica de probabilidad ni de graficación.

Uso (punto de entrada del proyecto)
────────────────────────────────────
    from src.ui.app import App
    App().mainloop()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
import threading

from src.ui.theme              import T
from src.ui.state              import AppState
from src.ui.controller         import BayesController
from src.ui.widgets            import StyledFrame, HeaderBar
from src.ui.panels.sidebar     import SidebarPanel
from src.ui.panels.charts      import ChartsPanel
from src.ui.panels.statusbar   import StatusBar


class App(tk.Tk):
    """
    Ventana principal de Weather Bayes.

    Layout
    ──────
    ┌─────────────────────────────────────────────────┐
    │  HeaderBar                                      │
    ├──────────────┬──────────────────────────────────┤
    │              │                                  │
    │  SidebarPanel│  ChartsPanel (Notebook)          │
    │  (controles) │  (9 pestañas de gráficas)        │
    │              │                                  │
    ├──────────────┴──────────────────────────────────┤
    │  StatusBar                                      │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()

        self.title("Weather Bayes — Análisis Bayesiano de Precipitación")
        self.configure(bg=T.bg)
        self.geometry("1400x820")
        self.minsize(900, 600)

        # Intentar pantalla completa en escritorio
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        # ── Estado y controller ───────────────────────────────────────────
        self._state      = AppState()
        self._controller = BayesController(self._state)

        # ── Construir layout ──────────────────────────────────────────────
        self._build_layout()

        # ── Cargar datos automáticamente al arrancar ──────────────────────
        self.after(100, self._load_data)

    # ── Construcción del layout ───────────────────────────────────────────

    def _build_layout(self) -> None:
        # — Header ─────────────────────────────────────────────────────────
        self._header = HeaderBar(self, title="🌦  Weather Bayes")
        self._header.pack(fill=tk.X, side=tk.TOP)

        # — Status bar (fija abajo antes del body para que pack funcione) ──
        self._statusbar = StatusBar(self)
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        # — Body: sidebar + charts ─────────────────────────────────────────
        body = StyledFrame(self, bg=T.bg)
        body.pack(fill=tk.BOTH, expand=True)

        self._sidebar = SidebarPanel(
            body,
            state            = self._state,
            on_compute       = self._on_compute,
            on_target_change = self._on_target_change,
        )
        self._sidebar.pack(fill=tk.Y, side=tk.LEFT)

        # Línea divisoria vertical
        tk.Frame(body, bg=T.surface2, width=1).pack(fill=tk.Y, side=tk.LEFT)

        self._charts = ChartsPanel(body)
        self._charts.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

    # ── Carga de datos ────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Carga el CSV en un hilo separado para no bloquear la UI."""
        self._statusbar.set_message("Cargando datos…", "info")
        self._statusbar.set_loading(True)

        def _worker():
            try:
                self._controller.load_data()
                self.after(0, self._on_data_loaded)
            except FileNotFoundError as exc:
                self.after(0, lambda: self._on_load_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_data_loaded(self) -> None:
        """Callback ejecutado en el hilo principal tras cargar datos."""
        self._statusbar.set_loading(False)
        self._statusbar.set_message(self._state.status_msg, "success")
        self._header.set_subtitle(
            f"{len(self._state.df)} registros  ·  "
            f"{self._state.df['Date'].min().date()} → "
            f"{self._state.df['Date'].max().date()}"
        )
        self._sidebar.refresh_combos()

    def _on_load_error(self, msg: str) -> None:
        self._statusbar.set_loading(False)
        self._statusbar.set_message(f"Error al cargar: {msg}", "error")
        messagebox.showerror(
            "Error de carga",
            f"No se pudo cargar el CSV:\n\n{msg}\n\n"
            "Verifica que weather.csv esté en src/data/",
        )

    # ── Callbacks de la sidebar ───────────────────────────────────────────

    def _on_target_change(self) -> None:
        """Sincroniza el target seleccionado con el estado."""
        self._sidebar.sync_target_to_state()

    def _on_compute(self) -> None:
        """
        Callback del botón ▶ Calcular.
        Ejecuta los cálculos en un hilo separado y luego actualiza las gráficas.
        """
        if not self._state.loaded:
            messagebox.showwarning(
                "Sin datos",
                "Los datos aún no se han cargado. Espera un momento.",
            )
            return

        self._statusbar.set_message("Calculando…", "info")
        self._statusbar.set_loading(True)
        self._sidebar.clear_results()

        def _worker():
            try:
                self._controller.compute()
                self.after(0, self._on_compute_done)
            except Exception as exc:
                self.after(0, lambda: self._on_compute_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_compute_done(self) -> None:
        """Callback ejecutado en el hilo principal tras completar el cómputo."""
        ctrl = self._controller

        # — Actualizar todas las gráficas en orden de pestañas ────────────
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
                # Una figura fallida no debe bloquear las demás
                print(f"[warn] figura '{tab_key}' falló: {exc}")

        # — Resultados numéricos en el sidebar ────────────────────────────
        lines = ctrl.build_results_lines()
        self._sidebar.update_results(lines)

        # — Activar la primera pestaña relevante ──────────────────────────
        self._charts.select_tab("posterior")

        # — Estado final ──────────────────────────────────────────────────
        self._statusbar.set_loading(False)
        self._statusbar.set_message(self._state.status_msg, "success")

    def _on_compute_error(self, msg: str) -> None:
        self._statusbar.set_loading(False)
        self._statusbar.set_message(f"Error en el cálculo: {msg}", "error")
        messagebox.showerror(
            "Error de cálculo",
            f"Ocurrió un error durante el análisis:\n\n{msg}",
        )