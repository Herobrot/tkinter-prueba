"""
naive_bayes.py
==============
Clasificador Naive Bayes y métricas de evaluación.

Responsabilidad única: dado un DataFrame, una columna objetivo binaria
y una lista de columnas de evidencia numéricas, implementar el clasificador
Naive Bayes gaussiano desde cero y calcular:

    · P(Fallo | x₁, x₂, …, xₙ)  — probabilidad posterior por muestra
    · Predicción binaria (clase 0 / 1)
    · Matriz de confusión
    · Accuracy, Sensibilidad (Recall), Especificidad

Sin dependencias de sklearn — solo numpy y pandas.
Los resultados son dataclasses inmutables listos para Tkinter/matplotlib/seaborn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Value Objects
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConfusionMatrix:
    """
    Matriz de confusión 2×2 para clasificación binaria.

    Convención
    ──────────
        Positivo = evento ocurre    (target == 1)
        Negativo = evento no ocurre (target == 0)

    Atributos
    ─────────
    tp : True  Positives — predijo 1, real 1
    fp : False Positives — predijo 1, real 0
    tn : True  Negatives — predijo 0, real 0
    fn : False Negatives — predijo 0, real 1
    """
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dataframe(self) -> pd.DataFrame:
        """
        DataFrame 2×2 con etiquetas.
        Listo para visualizar con seaborn.heatmap().
        """
        return pd.DataFrame(
            data    = [[self.tp, self.fp],
                       [self.fn, self.tn]],
            index   = ["Real: Positivo", "Real: Negativo"],
            columns = ["Pred: Positivo", "Pred: Negativo"],
        )

    def as_flat_series(self) -> pd.Series:
        """Serie plana para mostrar en tabla de la UI."""
        return pd.Series({"TP": self.tp, "FP": self.fp,
                          "TN": self.tn, "FN": self.fn})


@dataclass(frozen=True)
class ClassifierMetrics:
    """
    Métricas de evaluación del clasificador.

    Fórmulas
    ────────
    Accuracy     = (TP + TN) / (TP + TN + FP + FN)
    Sensibilidad = TP / (TP + FN)   — tasa de verdaderos positivos
    Especificidad= TN / (TN + FP)   — tasa de verdaderos negativos
    Precisión    = TP / (TP + FP)
    """
    accuracy:      float   # (TP + TN) / N
    sensitivity:   float   # TP / (TP + FN)   — Recall
    specificity:   float   # TN / (TN + FP)
    precision:     float   # TP / (TP + FP)
    confusion:     ConfusionMatrix

    def as_series(self) -> pd.Series:
        """
        Serie de métricas lista para barras horizontales en matplotlib.
        """
        return pd.Series(
            {
                "Accuracy":      self.accuracy,
                "Sensibilidad":  self.sensitivity,
                "Especificidad": self.specificity,
                "Precisión":     self.precision,
            },
            name="Métricas",
        )


@dataclass
class NaiveBayesResult:
    """
    Resultado completo del clasificador Naive Bayes.

    Atributos
    ─────────
    target          : columna objetivo
    features        : columnas usadas como predictoras
    y_true          : array con valores reales (0/1)
    y_pred          : array con predicciones  (0/1)
    y_prob          : array con P(clase=1 | x₁…xₙ) por muestra
    metrics         : ClassifierMetrics
    class_stats     : estadísticas μ y σ por clase y feature
    """
    target:      str
    features:    list[str]
    y_true:      np.ndarray
    y_pred:      np.ndarray
    y_prob:      np.ndarray
    metrics:     ClassifierMetrics
    class_stats: dict[int, pd.DataFrame]   # {0: df_stats_clase0, 1: df_stats_clase1}

    def prediction_dataframe(self) -> pd.DataFrame:
        """
        DataFrame con real, predicción y probabilidad posterior por fila.
        Útil para scatter plots de la UI.
        """
        return pd.DataFrame(
            {
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "p_positive": self.y_prob,
            }
        )


# ═════════════════════════════════════════════════════════════════════════════
# Clasificador
# ═════════════════════════════════════════════════════════════════════════════

class NaiveBayesClassifier:
    """
    Naive Bayes gaussiano implementado desde cero.

    Asume independencia condicional entre features dada la clase:
        P(Fallo | x₁,…,xₙ) ∝ P(Fallo) · ∏ P(xᵢ | Fallo)

    donde cada P(xᵢ | clase) se modela con una distribución normal:
        P(xᵢ | clase) = N(xᵢ; μ_{clase,i}, σ_{clase,i})

    Parámetros
    ──────────
    df       : DataFrame preprocesado (salida de DataPreprocessor)
    target   : columna binaria objetivo (0/1)
    features : columnas numéricas a usar como predictoras
    """

    def __init__(
        self,
        df:       pd.DataFrame,
        target:   str,
        features: list[str],
    ) -> None:
        self._validate(df, target, features)
        subset = df[[target] + features].dropna()
        self._df       = subset
        self._target   = target
        self._features = features

    # ── API pública ───────────────────────────────────────────────────────

    def run(self) -> NaiveBayesResult:
        """
        Entrena el modelo y evalúa sobre el mismo conjunto de datos.
        Devuelve NaiveBayesResult con predicciones y métricas completas.
        """
        class_stats = self._compute_class_statistics()
        priors      = self._compute_priors()
        y_true      = self._df[self._target].to_numpy(dtype=int)
        X           = self._df[self._features].to_numpy(dtype=float)

        y_prob  = self._predict_proba(X, class_stats, priors)
        y_pred  = (y_prob >= 0.5).astype(int)
        metrics = self._compute_metrics(y_true, y_pred)

        return NaiveBayesResult(
            target      = self._target,
            features    = self._features,
            y_true      = y_true,
            y_pred      = y_pred,
            y_prob      = y_prob,
            metrics     = metrics,
            class_stats = class_stats,
        )

    # ── Entrenamiento ─────────────────────────────────────────────────────

    def _compute_class_statistics(self) -> dict[int, pd.DataFrame]:
        """
        Calcula μ y σ de cada feature para cada clase.

        Devuelve:
            {
              0: DataFrame(index=features, columns=['mean','std']),
              1: DataFrame(index=features, columns=['mean','std']),
            }
        """
        stats = {}
        for cls in (0, 1):
            subset = self._df[self._df[self._target] == cls][self._features]
            stats[cls] = pd.DataFrame(
                {
                    "mean": subset.mean(),
                    "std":  subset.std().clip(lower=1e-9),  # evita σ = 0
                }
            )
        return stats

    def _compute_priors(self) -> dict[int, float]:
        """
        P(clase) para cada clase.
        """
        n = len(self._df)
        return {
            0: (self._df[self._target] == 0).sum() / n,
            1: (self._df[self._target] == 1).sum() / n,
        }

    # ── Predicción ────────────────────────────────────────────────────────

    def _predict_proba(
        self,
        X:           np.ndarray,
        class_stats: dict[int, pd.DataFrame],
        priors:      dict[int, float],
    ) -> np.ndarray:
        """
        Calcula P(clase=1 | x₁,…,xₙ) por muestra usando log-verosimilitud
        para evitar underflow numérico.

        Fórmula (en log)
        ────────────────
            log P(clase | x) = log P(clase)
                               + Σᵢ log N(xᵢ ; μ_{clase,i}, σ_{clase,i})

        La probabilidad posterior se obtiene con softmax sobre las dos clases.
        """
        log_posteriors = {}
        for cls in (0, 1):
            prior_val = priors.get(cls, 0.0)
            # Clase sin muestras → log-posterior = -inf (nunca se predice)
            if prior_val == 0.0 or cls not in class_stats:
                log_posteriors[cls] = np.full(X.shape[0], -np.inf)
                continue
            mu  = class_stats[cls]["mean"].to_numpy()
            sig = class_stats[cls]["std"].to_numpy()
            # Log-verosimilitud gaussiana para cada muestra: shape (n_samples,)
            log_likelihood = np.sum(
                -0.5 * ((X - mu) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi)),
                axis=1,
            )
            log_posteriors[cls] = np.log(prior_val + 1e-12) + log_likelihood

        # Softmax para convertir log-posteriors a probabilidades
        log0 = log_posteriors[0]
        log1 = log_posteriors[1]
        # Truco numérico: restar el máximo antes de exp
        max_log = np.maximum(log0, log1)
        exp0    = np.exp(log0 - max_log)
        exp1    = np.exp(log1 - max_log)
        return exp1 / (exp0 + exp1)   # P(clase=1 | x)

    # ── Métricas ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ClassifierMetrics:
        """
        Calcula la matriz de confusión y métricas derivadas.

        Fórmulas
        ────────
        TP = Σ (real=1 ∧ pred=1)
        FP = Σ (real=0 ∧ pred=1)
        TN = Σ (real=0 ∧ pred=0)
        FN = Σ (real=1 ∧ pred=0)

        Accuracy     = (TP + TN) / N
        Sensibilidad = TP / (TP + FN)
        Especificidad= TN / (TN + FP)
        Precisión    = TP / (TP + FP)
        """
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        n = tp + fp + tn + fn

        def safe(num: int, den: int) -> float:
            return round(num / den, 6) if den > 0 else 0.0

        return ClassifierMetrics(
            accuracy    = safe(tp + tn, n),
            sensitivity = safe(tp, tp + fn),
            specificity = safe(tn, tn + fp),
            precision   = safe(tp, tp + fp),
            confusion   = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn),
        )

    # ── Validación ────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame, target: str, features: list[str]) -> None:
        if target not in df.columns:
            raise KeyError(f"La columna objetivo '{target}' no existe.")
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise KeyError(f"Columnas de features no encontradas: {missing}")
        unique = set(df[target].dropna().unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"'{target}' debe ser binaria (0/1). Valores: {unique}"
            )
        if len(features) == 0:
            raise ValueError("Se requiere al menos una columna de features.")