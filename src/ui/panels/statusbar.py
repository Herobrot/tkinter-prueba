"""
src/ui/panels/statusbar.py
==========================
Responsabilidad única: barra de estado inferior que muestra el mensaje
de estado actual de la sesión (state.status_msg).

Expone set_message() y set_loading() para que app.py actualice el estado
visualmente sin acoplarse a los detalles del widget.
"""

from __future__ import annotations

import tkinter as tk
from src.ui.theme   import T
from src.ui.widgets import StyledFrame


class StatusBar(StyledFrame):
    """
    Barra de estado fija en la parte inferior de la ventana.
    Muestra un mensaje de texto y un indicador de carga opcional.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=T.panel, **kwargs)
        self.configure(height=T.statusbar_h)
        self.pack_propagate(False)

        # Indicador de actividad (punto animado)
        self._indicator = tk.Label(
            self, text="●",
            bg=T.panel, fg=T.disabled,
            font=T.font_xs,
        )
        self._indicator.pack(side=tk.LEFT, padx=(T.pad_sm, T.pad_xs))

        # Mensaje principal
        self._msg_var = tk.StringVar(value="Listo")
        tk.Label(
            self,
            textvariable=self._msg_var,
            bg=T.panel, fg=T.subtext,
            font=T.font_xs,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Versión / info derecha
        tk.Label(
            self, text="Weather Bayes v1.0",
            bg=T.panel, fg=T.disabled,
            font=T.font_xs,
        ).pack(side=tk.RIGHT, padx=T.pad_sm)

    def set_message(self, msg: str, level: str = "info") -> None:
        """
        Actualiza el texto del mensaje.
        level: "info" | "success" | "warning" | "error"
        """
        colors = {
            "info":    T.subtext,
            "success": T.green,
            "warning": T.yellow,
            "error":   T.red,
        }
        self._msg_var.set(msg)
        self._indicator.configure(fg=colors.get(level, T.subtext))

    def set_loading(self, active: bool) -> None:
        """Muestra/oculta el indicador de carga animado."""
        self._indicator.configure(fg=T.blue if active else T.disabled)