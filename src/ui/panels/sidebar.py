"""
src/ui/panels/sidebar.py
========================
Responsabilidad única: panel lateral izquierdo con todos los controles
de configuración del análisis.

Expone callbacks hacia afuera via on_compute y on_target_change.
No realiza cálculos; delega todo al controller via los callbacks
registrados desde app.py.

Controles
─────────
    · ComboRow  → Variable objetivo (target)
    · ComboRow  → Variable de evidencia (evidence)
    · SliderRow → Umbral (percentil 0–100)
    · StyledButton → ▶ Calcular
    · ResultsTable → resultados numéricos de la sesión
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from src.ui.theme   import T
from src.ui.widgets import (
    StyledFrame, SectionLabel, StyledButton,
    ComboRow, SliderRow, ResultsTable, Separator,
)
from src.ui.state   import AppState


class SidebarPanel(StyledFrame):
    """
    Panel lateral de controles.

    Parámetros
    ──────────
    parent          : widget padre
    state           : AppState compartido con el controller
    on_compute      : callback llamado cuando el usuario pulsa ▶ Calcular
    on_target_change: callback llamado cuando cambia la variable objetivo
    """

    def __init__(
        self,
        parent,
        state:            AppState,
        on_compute:       Callable,
        on_target_change: Callable,
        **kwargs,
    ):
        super().__init__(parent, bg=T.panel, **kwargs)
        self.configure(width=T.sidebar_width)
        self.pack_propagate(False)

        self._state            = state
        self._on_compute       = on_compute
        self._on_target_change = on_target_change

        self._build()

    # ── Construcción ──────────────────────────────────────────────────────

    def _build(self) -> None:
        s = self._state

        # — Logo / título del panel ────────────────────────────────────────
        title_frame = StyledFrame(self, bg=T.panel)
        title_frame.pack(fill=tk.X, padx=T.pad_md, pady=(T.pad_lg, T.pad_sm))

        tk.Label(
            title_frame, text="🌦  Weather Bayes",
            bg=T.panel, fg=T.text,
            font=T.font_lg,
        ).pack(anchor="w")
        tk.Label(
            title_frame, text="Análisis bayesiano de precipitación",
            bg=T.panel, fg=T.subtext,
            font=T.font_xs,
        ).pack(anchor="w")

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Sección: Variable ──────────────────────────────────────────────
        SectionLabel(self, "Variables").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        self._target_combo = ComboRow(
            self, "Variable objetivo",
            values=s.binary_cols or ["RainTomorrow"],
        )
        self._target_combo.pack(fill=tk.X)
        self._target_combo.var.trace_add(
            "write", lambda *_: self._on_target_change()
        )

        self._evidence_combo = ComboRow(
            self, "Variable de evidencia",
            values=s.numeric_cols or ["Humidity3pm"],
        )
        self._evidence_combo.pack(fill=tk.X)
        self._evidence_combo.var.trace_add(
            "write", lambda *_: self._sync_evidence()
        )

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Sección: Umbral ────────────────────────────────────────────────
        SectionLabel(self, "Umbral de evidencia").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        self._slider = SliderRow(self, "Percentil  →  valor umbral")
        self._slider.pack(fill=tk.X)
        self._slider.var.trace_add("write", lambda *_: self._sync_threshold())

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Botón Calcular ─────────────────────────────────────────────────
        self._btn_compute = StyledButton(
            self, text="▶   Calcular",
            command=self._on_compute,
            variant="success",
        )
        self._btn_compute.pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Sección: Resultados numéricos ──────────────────────────────────
        SectionLabel(self, "Resultados").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        self._results_table = ResultsTable(self)
        self._results_table.pack(
            fill=tk.BOTH, expand=True,
            padx=T.pad_md, pady=(T.pad_xs, T.pad_md)
        )

        # Scroll vertical
        scrollbar = tk.Scrollbar(
            self._results_table,
            command=self._results_table.yview,
            bg=T.surface, troughcolor=T.panel,
        )
        self._results_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ── API pública ───────────────────────────────────────────────────────

    def refresh_combos(self) -> None:
        """
        Actualiza los valores de los ComboBox con los datos del estado.
        Llamar después de cargar el DataFrame.
        """
        s = self._state
        self._target_combo.set_values(
            s.all_target_candidates, default=s.target
        )
        self._evidence_combo.set_values(
            s.numeric_cols, default=s.evidence
        )
        self._sync_threshold()

    def update_results(self, lines: list[tuple[str, str]]) -> None:
        """Escribe los resultados en el ResultsTable."""
        self._results_table.write(lines)

    def clear_results(self) -> None:
        self._results_table.clear()

    # ── Sincronización con AppState ───────────────────────────────────────

    def _sync_evidence(self) -> None:
        val = self._evidence_combo.get()
        if val:
            self._state.evidence = val
            self._sync_threshold()

    def _sync_threshold(self) -> None:
        """Calcula el valor real del umbral a partir del percentil."""
        import numpy as np
        s   = self._state
        pct = self._slider.get()
        s.threshold_pct = pct

        if s.df is not None and s.evidence in s.df.columns:
            val = float(np.percentile(s.df[s.evidence].dropna(), pct))
            s.threshold_val = val
            self._slider.set_display_value(val)
        else:
            self._slider.set_display_value(float(pct))

    def sync_target_to_state(self) -> None:
        val = self._target_combo.get()
        if val:
            self._state.target = val