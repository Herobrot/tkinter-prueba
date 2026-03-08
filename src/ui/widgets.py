"""
src/ui/widgets.py
=================
Responsabilidad única: proporcionar widgets Tkinter estilizados y
reutilizables que aplican el UITheme de forma consistente.

Ningún widget aquí contiene lógica de negocio ni de probabilidad.
Son componentes puramente visuales parametrizados.

Widgets disponibles
───────────────────
    StyledFrame     → Frame con fondo de tema
    HeaderBar       → barra de título de la aplicación
    SectionLabel    → etiqueta de sección con línea separadora
    StyledButton    → botón con estilos primario / secundario / danger
    ComboRow        → fila label + Combobox
    SliderRow       → fila label + Scale + valor numérico
    MetricCard      → tarjeta con nombre + valor + color de acento
    ResultsTable    → Text de solo lectura para resultados tabulados
    Separator       → línea horizontal de separación
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .theme import T


# ═════════════════════════════════════════════════════════════════════════════
# Frame base
# ═════════════════════════════════════════════════════════════════════════════

class StyledFrame(tk.Frame):
    """Frame con color de fondo del tema. Base de todos los paneles."""

    def __init__(self, parent, bg: str = T.bg, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Barra de encabezado
# ═════════════════════════════════════════════════════════════════════════════

class HeaderBar(StyledFrame):
    """
    Barra superior fija con título de la app y subtítulo de estado.
    El subtítulo se actualiza desde fuera via set_subtitle().
    """

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, bg=T.panel, **kwargs)
        self.configure(height=T.header_h)
        self.pack_propagate(False)

        tk.Label(
            self, text=title,
            bg=T.panel, fg=T.blue,
            font=T.font_xl,
        ).pack(side=tk.LEFT, padx=T.pad_lg)

        self._subtitle_var = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self._subtitle_var,
            bg=T.panel, fg=T.subtext,
            font=T.font_sm,
        ).pack(side=tk.LEFT, padx=T.pad_sm)

    def set_subtitle(self, text: str) -> None:
        self._subtitle_var.set(text)


# ═════════════════════════════════════════════════════════════════════════════
# Etiqueta de sección
# ═════════════════════════════════════════════════════════════════════════════

class SectionLabel(StyledFrame):
    """
    Etiqueta de sección con línea decorativa inferior.
    Usada para separar grupos de controles en el sidebar.
    """

    def __init__(self, parent, text: str, **kwargs):
        super().__init__(parent, bg=T.panel, **kwargs)

        tk.Label(
            self, text=text.upper(),
            bg=T.panel, fg=T.lavender,
            font=T.font_xs,
        ).pack(anchor="w")

        tk.Frame(self, bg=T.surface2, height=1).pack(fill=tk.X, pady=(2, 0))


# ═════════════════════════════════════════════════════════════════════════════
# Botón estilizado
# ═════════════════════════════════════════════════════════════════════════════

class StyledButton(tk.Button):
    """
    Botón con tres variantes visuales:
        "primary"   → fondo azul, texto oscuro
        "secondary" → fondo surface, texto claro
        "success"   → fondo verde, texto oscuro
        "danger"    → fondo rojo, texto oscuro
    """

    _VARIANTS: dict[str, tuple[str, str]] = {
        "primary":   (T.blue,    T.bg),
        "secondary": (T.surface, T.text),
        "success":   (T.green,   T.bg),
        "danger":    (T.red,     T.bg),
    }

    def __init__(
        self,
        parent,
        text: str,
        command: Callable = None,
        variant: str = "primary",
        **kwargs,
    ):
        bg, fg = self._VARIANTS.get(variant, self._VARIANTS["primary"])
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg, fg=fg,
            font=T.font_md,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground=T.surface2,
            activeforeground=T.text,
            padx=T.pad_md,
            pady=T.pad_xs,
            **kwargs,
        )

    def set_state(self, enabled: bool) -> None:
        self.configure(state=tk.NORMAL if enabled else tk.DISABLED)


# ═════════════════════════════════════════════════════════════════════════════
# Fila: Etiqueta + Combobox
# ═════════════════════════════════════════════════════════════════════════════

class ComboRow(StyledFrame):
    """
    Fila compacta con una etiqueta descriptiva y un Combobox.
    Expone .var (StringVar) y .set_values() para actualizar opciones.
    """

    def __init__(
        self,
        parent,
        label: str,
        values: list[str] = (),
        **kwargs,
    ):
        super().__init__(parent, bg=T.panel, **kwargs)

        tk.Label(
            self, text=label,
            bg=T.panel, fg=T.subtext,
            font=T.font_sm,
        ).pack(anchor="w", padx=T.pad_md, pady=(T.pad_sm, 2))

        self.var = tk.StringVar()

        # Estilo ttk para Combobox oscuro
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox",
            fieldbackground=T.surface,
            background=T.surface,
            foreground=T.text,
            selectbackground=T.blue,
            selectforeground=T.bg,
            bordercolor=T.surface2,
            arrowcolor=T.subtext,
        )

        self._combo = ttk.Combobox(
            self,
            textvariable=self.var,
            values=list(values),
            state="readonly",
            style="Dark.TCombobox",
            width=28,
        )
        self._combo.pack(padx=T.pad_md, pady=(0, T.pad_sm))

    def set_values(self, values: list[str], default: str = "") -> None:
        self._combo.configure(values=values, state="readonly")
        self.var.set(default if default in values else (values[0] if values else ""))

    def get(self) -> str:
        return self.var.get()


# ═════════════════════════════════════════════════════════════════════════════
# Fila: Etiqueta + Slider + valor numérico
# ═════════════════════════════════════════════════════════════════════════════

class SliderRow(StyledFrame):
    """
    Fila con etiqueta, slider horizontal y etiqueta del valor actual.
    El valor real (percentil → umbral) se calcula externamente y se muestra
    via set_display_value().
    """

    def __init__(
        self,
        parent,
        label: str,
        from_: int = 0,
        to: int = 100,
        initial: int = 50,
        **kwargs,
    ):
        super().__init__(parent, bg=T.panel, **kwargs)

        header = StyledFrame(self, bg=T.panel)
        header.pack(fill=tk.X, padx=T.pad_md, pady=(T.pad_sm, 2))

        tk.Label(
            header, text=label,
            bg=T.panel, fg=T.subtext,
            font=T.font_sm,
        ).pack(side=tk.LEFT)

        self._display_var = tk.StringVar(value=f"{initial}")
        tk.Label(
            header, textvariable=self._display_var,
            bg=T.panel, fg=T.yellow,
            font=T.font_code,
        ).pack(side=tk.RIGHT)

        self.var = tk.IntVar(value=initial)
        tk.Scale(
            self,
            variable=self.var,
            from_=from_, to=to,
            orient=tk.HORIZONTAL,
            bg=T.panel, fg=T.text,
            troughcolor=T.surface,
            activebackground=T.blue,
            highlightthickness=0,
            showvalue=False,
            length=260,
        ).pack(padx=T.pad_md, pady=(0, T.pad_sm))

    def get(self) -> int:
        return self.var.get()

    def set_display_value(self, value: float, unit: str = "") -> None:
        self._display_var.set(f"{value:.2f}{unit}")


# ═════════════════════════════════════════════════════════════════════════════
# Tarjeta de métrica
# ═════════════════════════════════════════════════════════════════════════════

class MetricCard(StyledFrame):
    """
    Tarjeta compacta que muestra un nombre y un valor numérico destacado.
    El color del valor cambia según el nivel: verde ≥ 0.75, amarillo ≥ 0.5, rojo.
    """

    def __init__(self, parent, name: str, **kwargs):
        super().__init__(parent, bg=T.surface, **kwargs)
        self.configure(padx=T.pad_md, pady=T.pad_sm)

        tk.Label(
            self, text=name,
            bg=T.surface, fg=T.subtext,
            font=T.font_xs,
        ).pack(anchor="w")

        self._value_var = tk.StringVar(value="—")
        self._value_lbl = tk.Label(
            self, textvariable=self._value_var,
            bg=T.surface, fg=T.text,
            font=("Segoe UI", 18, "bold"),
        )
        self._value_lbl.pack(anchor="w")

    def set_value(self, value: float) -> None:
        self._value_var.set(f"{value:.4f}")
        color = T.green if value >= 0.75 else (T.yellow if value >= 0.5 else T.red)
        self._value_lbl.configure(fg=color)

    def reset(self) -> None:
        self._value_var.set("—")
        self._value_lbl.configure(fg=T.disabled)


# ═════════════════════════════════════════════════════════════════════════════
# Tabla de resultados (texto monoespaciado de solo lectura)
# ═════════════════════════════════════════════════════════════════════════════

class ResultsTable(tk.Text):
    """
    Widget Text configurado como visor de solo lectura para resultados
    tabulados de probabilidades.
    Soporta coloreado de líneas via tag_highlight().
    """

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=T.surface,
            fg=T.text,
            font=T.font_code,
            relief=tk.FLAT,
            wrap=tk.NONE,
            state=tk.DISABLED,
            cursor="arrow",
            padx=T.pad_sm,
            pady=T.pad_sm,
            **kwargs,
        )
        # Tags de color
        self.tag_configure("header",  foreground=T.lavender, font=(*T.font_code, "bold"))
        self.tag_configure("value",   foreground=T.blue)
        self.tag_configure("high",    foreground=T.red)
        self.tag_configure("neutral", foreground=T.subtext)

    def write(self, lines: list[tuple[str, str]]) -> None:
        """
        Escribe líneas con tags.
        lines = [("texto", "tag"), ...]  tag puede ser: header/value/high/neutral/""
        """
        self.configure(state=tk.NORMAL)
        self.delete("1.0", tk.END)
        for text, tag in lines:
            if tag:
                self.insert(tk.END, text + "\n", tag)
            else:
                self.insert(tk.END, text + "\n")
        self.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self.configure(state=tk.NORMAL)
        self.delete("1.0", tk.END)
        self.configure(state=tk.DISABLED)


# ═════════════════════════════════════════════════════════════════════════════
# Separador horizontal
# ═════════════════════════════════════════════════════════════════════════════

class Separator(tk.Frame):
    """Línea horizontal decorativa de separación."""

    def __init__(self, parent, color: str = T.surface2, **kwargs):
        super().__init__(parent, bg=color, height=1, **kwargs)