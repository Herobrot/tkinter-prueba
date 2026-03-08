"""
src/ui/theme.py
===============
Responsabilidad única: centralizar todas las constantes visuales de la
interfaz Tkinter (colores, fuentes, espaciados).

Espeja intencionalmente la Palette de src/graphs/style.py para mantener
coherencia visual entre los widgets Tk y las figuras de matplotlib.

Ningún otro módulo de UI define colores o fuentes directamente;
todos importan desde aquí.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class UITheme:
    # ── Fondos ────────────────────────────────────────────────────────────
    bg:         str = "#1e1e2e"   # ventana principal
    panel:      str = "#181825"   # sidebars, cabeceras
    surface:    str = "#313244"   # cards, inputs, separadores
    surface2:   str = "#45475a"   # hover, bordes sutiles

    # ── Texto ─────────────────────────────────────────────────────────────
    text:       str = "#cdd6f4"   # texto principal
    subtext:    str = "#a6adc8"   # etiquetas, placeholders
    disabled:   str = "#585b70"   # controles inactivos

    # ── Acentos (coherentes con graphs/style.py → PALETTE) ────────────────
    blue:       str = "#89b4fa"   # botones primarios, selección activa
    red:        str = "#f38ba8"   # alertas, P(Fallo)
    green:      str = "#a6e3a1"   # éxito, accuracy alta
    yellow:     str = "#f9e2af"   # advertencias, umbral
    lavender:   str = "#b4befe"   # acento secundario
    teal:       str = "#94e2d5"   # terciario

    # ── Tipografía ────────────────────────────────────────────────────────
    font_family:    str = "Segoe UI"
    font_mono:      str = "Cascadia Code"

    font_xs:    tuple = ("Segoe UI", 8)
    font_sm:    tuple = ("Segoe UI", 9)
    font_md:    tuple = ("Segoe UI", 10)
    font_lg:    tuple = ("Segoe UI", 12, "bold")
    font_xl:    tuple = ("Segoe UI", 14, "bold")
    font_code:  tuple = ("Cascadia Code", 9)

    # ── Espaciados ────────────────────────────────────────────────────────
    pad_xs:  int = 4
    pad_sm:  int = 8
    pad_md:  int = 12
    pad_lg:  int = 16
    pad_xl:  int = 24

    # ── Geometría de paneles ──────────────────────────────────────────────
    sidebar_width:  int = 300
    statusbar_h:    int = 24
    header_h:       int = 48


# Instancia global importable por todos los módulos UI
T = UITheme()