
"""
Практическая работа №2.
Модуль обнаружения аномалий временного ряда.

Методы:
- IQR
- Z-score
- Isolation Forest
- Local Outlier Factor
- STL
- Seasonal Hybrid ESD (приближённая реализация)
"""

import numpy as np
import pandas as pd

from scipy.stats import t as student_t
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


def detect_iqr(values, multiplier=1.5):
    """Обнаружение выбросов по межквартильному размаху."""
    s = pd.Series(values, dtype=float)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1

    if iqr == 0:
        return pd.Series(False, index=s.index)

    return (s < q1 - multiplier * iqr) | (
        s > q3 + multiplier * iqr
    )


def detect_zscore(values, threshold=3.0):
    """Обнаружение выбросов по стандартному Z-score."""
    s = pd.Series(values, dtype=float)
    std = s.std(ddof=0)

    if not np.isfinite(std) or std == 0:
        return pd.Series(False, index=s.index)

    z = (s - s.mean()).abs() / std
    return z > threshold


def detect_isolation_forest(
    features,
    contamination=0.03,
    random_state=42
):
    """Обнаружение аномалий методом Isolation Forest."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    labels = model.fit_predict(features)
    return labels == -1


def detect_lof(features, contamination=0.03, n_neighbors=20):
    """Обнаружение локальных аномалий методом LOF."""
    n_neighbors = min(n_neighbors, len(features) - 1)

    if n_neighbors < 2:
        raise ValueError("Для LOF требуется минимум 3 наблюдения")

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    labels = model.fit_predict(features)
    return labels == -1


def stl_decompose(values, period=365):
    """Декомпозиция ряда на тренд, сезонность и остатки."""
    s = pd.Series(values, dtype=float)

    if s.isna().any():
        raise ValueError("STL не допускает пропуски в ряду")

    result = STL(
        s,
        period=period,
        robust=True
    ).fit()

    return result


def detect_stl(values, period=365, threshold=3.5):
    """Поиск контекстных аномалий в остатках STL."""
    result = stl_decompose(values, period=period)

    residuals = pd.Series(result.resid)
    median = residuals.median()
    mad = np.median(np.abs(residuals - median))

    if mad <= 1e-12:
        flags = pd.Series(False, index=residuals.index)
    else:
        robust_z = (
            0.6745 * (residuals - median).abs() / mad
        )
        flags = robust_z > threshold

    return flags, result


def generalized_esd(values, max_outliers=30, alpha=0.05):
    """
    Обобщённый ESD-тест Рознера.

    Возвращает индексы обнаруженных выбросов.
    """
    x = np.asarray(values, dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("ESD требует конечные числовые значения")

    n = len(x)
    max_outliers = min(max_outliers, n - 3)

    if max_outliers < 1:
        return np.array([], dtype=int)

    remaining = np.arange(n)
    removed = []
    statistics = []
    critical_values = []

    for i in range(1, max_outliers + 1):
        current = x[remaining]

        center = current.mean()
        std = current.std(ddof=1)

        if std <= 1e-12:
            break

        deviations = np.abs(current - center)
        local_index = int(np.argmax(deviations))

        statistic = deviations[local_index] / std

        current_n = len(current)
        p = 1 - alpha / (2 * current_n)
        t_value = student_t.ppf(p, current_n - 2)

        critical = (
            (current_n - 1) * t_value
            / np.sqrt(
                (current_n - 2 + t_value**2) * current_n
            )
        )

        removed.append(remaining[local_index])
        statistics.append(statistic)
        critical_values.append(critical)

        remaining = np.delete(remaining, local_index)

    significant = [
        i for i, (stat, critical)
        in enumerate(zip(statistics, critical_values))
        if stat > critical
    ]

    if not significant:
        return np.array([], dtype=int)

    k = max(significant) + 1
    return np.array(removed[:k], dtype=int)


def detect_seasonal_esd(
    values,
    period=365,
    max_anomaly_fraction=0.05,
    alpha=0.05
):
    """
    Сезонная декомпозиция + Generalized ESD.

    Учебная приближённая реализация Seasonal Hybrid ESD:
    ESD применяется к остаткам STL после удаления
    сезонной и трендовой компонент.

    Это не полная оригинальная реализация S-H-ESD.
    """
    result = stl_decompose(values, period=period)
    residuals = np.asarray(result.resid, dtype=float)

    max_outliers = max(
        1,
        int(len(residuals) * max_anomaly_fraction)
    )

    indices = generalized_esd(
        residuals,
        max_outliers=max_outliers,
        alpha=alpha
    )

    flags = np.zeros(len(residuals), dtype=bool)
    flags[indices] = True

    return pd.Series(flags), result


def run_anomaly_detection(
    df,
    target="temperature_2m_mean",
    date_col="time",
    period=365,
    contamination=0.03,
    random_state=42
):
    """
    Запускает все методы обнаружения аномалий.

    Возвращает:
        result_df: исходные данные с флагами
        summary_df: сводная таблица методов
        decomposition: результат STL
    """
    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    y = pd.to_numeric(data[target], errors="coerce")

    if y.isna().any():
        raise ValueError("Целевая переменная содержит пропуски")

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Целевая переменная содержит inf")

    # Статистические методы
    data["anomaly_iqr"] = detect_iqr(y).to_numpy()
    data["anomaly_zscore"] = detect_zscore(y).to_numpy()

    # Признаки для машинных методов:
    # температура и её прошлые значения
    ml_features = pd.DataFrame({
        "temperature": y,
        "lag_1": y.shift(1),
        "lag_7": y.shift(7),
        "rolling_mean_7": y.shift(1).rolling(7).mean(),
        "rolling_std_7": y.shift(1).rolling(7).std()
    })

    ml_features = ml_features.bfill().ffill()

    scaler = StandardScaler()
    X = scaler.fit_transform(ml_features)

    data["anomaly_isolation_forest"] = (
        detect_isolation_forest(
            X,
            contamination=contamination,
            random_state=random_state
        )
    )

    data["anomaly_lof"] = detect_lof(
        X,
        contamination=contamination
    )

    # Контекстные методы
    stl_flags, decomposition = detect_stl(
        y,
        period=period
    )

    data["anomaly_stl"] = stl_flags.to_numpy()

    esd_flags, _ = detect_seasonal_esd(
        y,
        period=period
    )

    data["anomaly_seasonal_esd"] = esd_flags.to_numpy()

    point_columns = [
        "anomaly_iqr",
        "anomaly_zscore",
        "anomaly_isolation_forest",
        "anomaly_lof"
    ]

    context_columns = [
        "anomaly_stl",
        "anomaly_seasonal_esd"
    ]

    data["point_anomaly"] = data[point_columns].any(axis=1)
    data["context_anomaly"] = data[context_columns].any(axis=1)

    data["anomaly_type"] = np.select(
        [
            data["point_anomaly"] & data["context_anomaly"],
            data["context_anomaly"],
            data["point_anomaly"]
        ],
        [
            "Точечная и контекстная",
            "Контекстная",
            "Точечная"
        ],
        default="Норма"
    )

    all_columns = point_columns + context_columns

    data["anomaly_votes"] = data[all_columns].sum(axis=1)
    data["is_anomaly"] = data["anomaly_votes"] > 0

    summary = []

    for column in all_columns:
        count = int(data[column].sum())

        summary.append({
            "method": column.replace("anomaly_", ""),
            "anomaly_count": count,
            "anomaly_percent": round(
                count / len(data) * 100, 2
            )
        })

    summary_df = pd.DataFrame(summary)

    return data, summary_df, decomposition
