from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_DEFAULT_CSV: Path = Path(__file__).parent / "data" / "weather.csv"

@dataclass(frozen=True)
class WeatherSchema:
    """
    Fuente de verdad del esquema de weather.csv.
    Las columnas están clasificadas de forma explícita; no se infieren.

    Expone tuplas de solo lectura que los módulos consumidores
    (UI, analizador bayesiano, visualizaciones) pueden importar directamente.
    """

    # Columnas tal como aparecen en el CSV —————————————————————————————————
    numeric_cols: tuple[str, ...] = (
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm",
        "Cloud9am", "Cloud3pm",
        "Temp9am", "Temp3pm",
        "RISK_MM",
    )

    categorical_cols: tuple[str, ...] = (
        "WindGustDir",
        "WindDir9am",
        "WindDir3pm",
    )

    binary_cols: tuple[str, ...] = (
        "RainToday",
        "RainTomorrow",
    )

    # Columna generada en el pipeline ——————————————————————————————————————
    datetime_col: str = "Date"

    # Valores que se mapean a 1 en columnas binarias ———————————————————————
    _POSITIVE_VALUES: frozenset = field(
        default_factory=lambda: frozenset({"yes", "Yes", "sí", "si", "1", "true", "verdadero"})
    )

    def all_source_cols(self) -> list[str]:
        """Columnas originales del CSV (sin la columna Date generada)."""
        return list(self.numeric_cols) + list(self.categorical_cols) + list(self.binary_cols)

    def is_positive_binary(self, value: str) -> bool:
        return str(value).strip().lower() in self._POSITIVE_VALUES


# Instancia global — importable directamente por otros módulos
WEATHER_SCHEMA = WeatherSchema()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — RESULTADO DEL PIPELINE  (Value Object)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PreprocessResult:
    """
    Objeto de transferencia que DataPreprocessor entrega a los consumidores.

    df     : DataFrame limpio y tipado con columna Date al frente
    schema : referencia al WeatherSchema usado — permite introspección
    """
    df:     pd.DataFrame
    schema: WeatherSchema

    def __post_init__(self):
        if self.schema.datetime_col not in self.df.columns:
            raise ValueError(
                f"La columna '{self.schema.datetime_col}' no está presente en el DataFrame."
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — INTERFACES  (I — contratos mínimos e independientes)
# ═════════════════════════════════════════════════════════════════════════════

class IDataLoader(ABC):
    """Contrato para cargar el DataFrame crudo desde cualquier fuente."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


class ITransformStep(ABC):
    """
    Contrato para un paso de transformación del pipeline.
    Recibe el DataFrame y el schema; devuelve el DataFrame modificado.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame, schema: WeatherSchema) -> pd.DataFrame:
        ...


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CARGA DE DATOS  (S — solo lee el archivo)
# ═════════════════════════════════════════════════════════════════════════════

class CsvDataLoader(IDataLoader):
    """
    Lee weather.csv y devuelve el DataFrame crudo sin modificaciones.
    Única responsabilidad: I/O de disco.
    """

    def __init__(self, filepath: str | Path = _DEFAULT_CSV) -> None:
        self._path = Path(filepath)

    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset en: {self._path}\n"
                "Coloca weather.csv dentro de la carpeta 'data/'."
            )
        return pd.read_csv(self._path)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — PASOS DE TRANSFORMACIÓN  (S — uno por responsabilidad)
# ═════════════════════════════════════════════════════════════════════════════

class NumericCaster(ITransformStep):
    """
    Castea las columnas numéricas declaradas en el schema a float64.
    Valores no convertibles quedan como NaN.
    Única responsabilidad: coerción numérica.
    """

    def apply(self, df: pd.DataFrame, schema: WeatherSchema) -> pd.DataFrame:
        for col in schema.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df


class BinaryEncoder(ITransformStep):
    """
    Codifica las columnas binarias declaradas (Yes/No → 1/0) a int8.
    Única responsabilidad: mapeo de valores binarios textuales a enteros.
    """

    def apply(self, df: pd.DataFrame, schema: WeatherSchema) -> pd.DataFrame:
        for col in schema.binary_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .map(lambda v: int(schema.is_positive_binary(v)))
                    .astype("int8")
                )
        return df


class TimestampBuilder(ITransformStep):
    """
    Genera la columna 'Date' asignando una fecha consecutiva por fila,
    comenzando en start_date (por defecto: 2026-01-01).

    Cada fila representa un día, por lo que el delta entre fechas es 1 día.
    Única responsabilidad: construir la serie temporal del dataset.
    """

    def __init__(self, start_date: str = "2026-01-01") -> None:
        self._start = pd.Timestamp(start_date)

    def apply(self, df: pd.DataFrame, schema: WeatherSchema) -> pd.DataFrame:
        dates = pd.date_range(start=self._start, periods=len(df), freq="D")
        df.insert(0, schema.datetime_col, dates)
        return df


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — ORQUESTADOR  (D — depende de abstracciones, no de concretos)
# ═════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Orquesta el pipeline: carga → pasos de transformación → resultado.

    Los pasos se ejecutan en el orden en que se pasan al constructor,
    lo que permite añadir, quitar o reordenar sin modificar esta clase.

    Parámetros
    ──────────
    loader  : implementación de IDataLoader
    steps   : lista ordenada de ITransformStep
    schema  : WeatherSchema con el esquema declarativo
    """

    def __init__(
        self,
        loader: IDataLoader          = None,
        steps:  list[ITransformStep] = None,
        schema: WeatherSchema        = None,
    ) -> None:
        self._loader = loader or CsvDataLoader()
        self._schema = schema or WEATHER_SCHEMA
        self._steps  = steps  or [
            NumericCaster(),      # 1.º castear numéricos
            BinaryEncoder(),      # 2.º codificar binarios
            TimestampBuilder(),   # 3.º insertar columna Date
        ]

    def run(self) -> PreprocessResult:
        """
        Ejecuta el pipeline completo.

        1. Carga el CSV crudo.
        2. Aplica cada ITransformStep en orden.
        3. Devuelve PreprocessResult(df, schema).
        """
        df = self._loader.load()

        for step in self._steps:
            df = step.apply(df, self._schema)

        return PreprocessResult(df=df, schema=self._schema)