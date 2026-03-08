"""
src/ui/panels/sidebar.py
========================
Responsabilidad única: panel lateral izquierdo con todos los controles
de configuración del análisis.

Cambios respecto a la versión anterior
───────────────────────────────────────
· Botón "📂 Abrir CSV" → abre filedialog y dispara carga + análisis automático
· Al cargar un archivo, el análisis se ejecuta automáticamente
· La lista de evidencias se filtra dinámicamente según la variable objetivo:
    - Se excluye siempre el target de la lista de evidencias
    - Solo se muestran columnas numéricas (compatibles con umbral)
· El botón ▶ Recalcular permite repetir el análisis con otra variable

Callbacks hacia app.py
──────────────────────
    on_file_selected(filepath) → usuario eligió un CSV
    on_compute()               → usuario pidió recalcular
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from typing import Callable

from src.ui.theme   import T
from src.ui.widgets import (
    StyledFrame, SectionLabel, StyledButton,
    ComboRow, SliderRow, ResultsTable, Separator,
)
from src.ui.state import AppState


class SidebarPanel(StyledFrame):
    """
    Panel lateral de controles.

    Parámetros
    ──────────
    parent            : widget padre
    state             : AppState compartido con el controller
    on_file_selected  : callback(filepath: str) — usuario eligió un CSV
    on_compute        : callback() — usuario pidió recalcular
    """

    def __init__(
        self,
        parent,
        state:             AppState,
        on_file_selected:  Callable[[str], None],
        on_compute:        Callable[[], None],
        **kwargs,
    ):
        super().__init__(parent, bg=T.panel, **kwargs)
        self.configure(width=T.sidebar_width)
        self.pack_propagate(False)

        self._state            = state
        self._on_file_selected = on_file_selected
        self._on_compute       = on_compute

        # Flag interno para evitar que los traces disparen recálculos
        # mientras se están actualizando los combos programáticamente
        self._updating_combos = False

        self._build()

    # ── Construcción ──────────────────────────────────────────────────────

    def _build(self) -> None:

        # — Título ─────────────────────────────────────────────────────────
        title_frame = StyledFrame(self, bg=T.panel)
        title_frame.pack(fill=tk.X, padx=T.pad_md, pady=(T.pad_lg, T.pad_sm))

        tk.Label(
            title_frame, text="🌦  Weather Bayes",
            bg=T.panel, fg=T.text,
            font=T.font_lg,
        ).pack(anchor="w")
        tk.Label(
            title_frame, text="Análisis bayesiano de variables",
            bg=T.panel, fg=T.subtext,
            font=T.font_xs,
        ).pack(anchor="w")

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Sección: Archivo CSV ───────────────────────────────────────────
        SectionLabel(self, "Datos").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        # Nombre del archivo cargado
        self._file_var = tk.StringVar(value="Ningún archivo seleccionado")
        tk.Label(
            self,
            textvariable=self._file_var,
            bg=T.panel, fg=T.subtext,
            font=T.font_xs,
            wraplength=T.sidebar_width - T.pad_xl,
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=T.pad_md, pady=(T.pad_xs, 0))

        self._btn_open = StyledButton(
            self,
            text="📂  Abrir CSV",
            command=self._open_file_dialog,
            variant="primary",
        )
        self._btn_open.pack(fill=tk.X, padx=T.pad_md, pady=(T.pad_xs, T.pad_sm))

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Sección: Variables ─────────────────────────────────────────────
        SectionLabel(self, "Variables").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        self._target_combo = ComboRow(
            self, "Variable objetivo",
            values=[],
        )
        self._target_combo.pack(fill=tk.X)
        # Al cambiar el target → filtrar evidencias y recalcular
        self._target_combo.var.trace_add(
            "write", lambda *_: self._on_target_changed()
        )

        self._evidence_combo = ComboRow(
            self, "Variable de evidencia",
            values=[],
        )
        self._evidence_combo.pack(fill=tk.X)
        self._evidence_combo.var.trace_add(
            "write", lambda *_: self._on_evidence_changed()
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

        # — Botón Recalcular ───────────────────────────────────────────────
        self._btn_compute = StyledButton(
            self, text="▶   Recalcular",
            command=self._on_compute,
            variant="success",
        )
        self._btn_compute.pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)
        self._btn_compute.set_state(False)   # deshabilitado hasta cargar datos

        Separator(self).pack(fill=tk.X, padx=T.pad_md, pady=T.pad_sm)

        # — Resultados numéricos ───────────────────────────────────────────
        SectionLabel(self, "Resultados").pack(
            fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 0)
        )

        self._results_table = ResultsTable(self)
        self._results_table.pack(
            fill=tk.BOTH, expand=True,
            padx=T.pad_md, pady=(T.pad_xs, T.pad_md),
        )

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
        Actualiza ambos ComboBox con las columnas del DataFrame cargado.
        Llamar después de que el controller haya cargado los datos.

        Aplica el filtrado de compatibilidad desde el primer momento:
        la lista de evidencias ya excluye el target actual y las categóricas.
        """
        self._updating_combos = True
        s = self._state

        # Target: todas las candidatas (binarias + numéricas)
        self._target_combo.set_values(
            s.all_target_candidates, default=s.target
        )

        # Evidencia: solo numéricas compatibles con el target actual
        evidence_opts = s.compatible_evidence_cols(s.target)
        default_ev = s.evidence if s.evidence in evidence_opts else (
            evidence_opts[0] if evidence_opts else ""
        )
        self._evidence_combo.set_values(evidence_opts, default=default_ev)
        s.evidence = default_ev

        self._sync_threshold()
        self._btn_compute.set_state(True)
        self._updating_combos = False

    def update_results(self, lines: list[tuple[str, str]]) -> None:
        self._results_table.write(lines)

    def clear_results(self) -> None:
        self._results_table.clear()

    def set_file_label(self, filename: str) -> None:
        """Muestra el nombre del archivo cargado bajo el botón."""
        self._file_var.set(f"📄  {filename}")

    # ── Callbacks internos ────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        """
        Abre el diálogo de selección de archivo.
        Si el usuario elige un CSV válido, llama a on_file_selected
        para que app.py coordine la carga y el análisis automático.
        """
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
        )
        if filepath:
            self._on_file_selected(filepath)

    def _on_target_changed(self) -> None:
        """
        Cuando el usuario cambia el target:
          1. Sincroniza con el estado
          2. Actualiza la lista de evidencias filtrando incompatibles
        El recálculo solo ocurre al pulsar ▶ Recalcular.
        """
        if self._updating_combos:
            return

        val = self._target_combo.get()
        if not val:
            return

        self._state.target = val

        # Reconstruir lista de evidencias excluyendo el nuevo target
        self._updating_combos = True
        s = self._state
        evidence_opts = s.compatible_evidence_cols(val)
        current_ev    = self._evidence_combo.get()

        # Mantener la evidencia actual si sigue siendo compatible,
        # si no, elegir la primera de la lista
        default_ev = current_ev if current_ev in evidence_opts else (
            evidence_opts[0] if evidence_opts else ""
        )
        self._evidence_combo.set_values(evidence_opts, default=default_ev)
        s.evidence = default_ev
        self._updating_combos = False

        self._sync_threshold()

    def _on_evidence_changed(self) -> None:
        """
        Cuando el usuario cambia la evidencia sincroniza el estado
        y actualiza el umbral mostrado.
        El recálculo solo ocurre al pulsar ▶ Recalcular.
        """
        if self._updating_combos:
            return

        val = self._evidence_combo.get()
        if not val:
            return

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

    def set_controls_enabled(self, enabled: bool) -> None:
        """
        Habilita o deshabilita todos los controles interactivos.
        Llamar con False al iniciar un cálculo y con True al terminar,
        para evitar que el usuario modifique selecciones mientras procesa.
        """
        self._btn_open.set_state(enabled)
        self._target_combo.set_enabled(enabled)
        self._evidence_combo.set_enabled(enabled)
        self._slider.set_enabled(enabled)
        self._btn_compute.set_state(enabled and self._state.loaded)