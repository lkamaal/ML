
"""Анализ мультиколлинеарности экзогенных признаков."""

import numpy as np
import pandas as pd


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет VIF для числовых признаков.

    Parameters
    ----------
    X : pd.DataFrame
        Матрица экзогенных переменных без пропусков.
        Постоянные столбцы недопустимы.

    Returns
    -------
    pd.DataFrame
        Таблица feature, VIF, отсортированная по убыванию VIF.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X должен быть pandas DataFrame.")

    if X.empty or X.shape[1] == 0:
        raise ValueError("Матрица признаков пуста.")

    if X.isna().any().any():
        raise ValueError("Матрица содержит пропуски.")

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Матрица содержит inf или NaN.")

    std = values.std(axis=0)

    if np.any(std < 1e-12):
        raise ValueError("Обнаружены постоянные признаки.")

    if X.shape[1] == 1:
        return pd.DataFrame({
            "feature": X.columns,
            "VIF": [1.0]
        })

    # Центрирование и стандартизация.
    Z = (values - values.mean(axis=0)) / std

    # Корреляционная матрица.
    corr = Z.T @ Z / len(Z)

    # VIF равен диагонали обратной корреляционной матрицы.
    # При точной линейной зависимости используем inf.
    try:
        eigenvalues = np.linalg.eigvalsh(corr)
        if eigenvalues.min() <= 1e-10:
            vif_values = _vif_with_collinearity(Z)
        else:
            vif_values = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif_values = _vif_with_collinearity(Z)

    result = pd.DataFrame({
        "feature": X.columns,
        "VIF": np.maximum(vif_values, 1.0)
    })

    return result.sort_values(
        "VIF", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _vif_with_collinearity(Z: np.ndarray) -> np.ndarray:
    """Рассчитывает VIF через вспомогательные регрессии."""
    n_features = Z.shape[1]
    result = np.zeros(n_features)

    for j in range(n_features):
        target = Z[:, j]
        others = np.delete(Z, j, axis=1)

        coefficients = np.linalg.lstsq(
            others, target, rcond=None
        )[0]

        residuals = target - others @ coefficients

        r2 = 1.0 - (
            np.sum(residuals ** 2) /
            np.sum(target ** 2)
        )

        result[j] = (
            np.inf if r2 >= 1 - 1e-10
            else 1.0 / max(1.0 - r2, 1e-12)
        )

    return result


def iterative_vif_filter(
    X: pd.DataFrame,
    threshold: float = 10.0,
    verbose: bool = True
):
    """
    Итеративно исключает признаки с VIF выше порога.

    Parameters
    ----------
    X : pd.DataFrame
        Числовая матрица признаков без пропусков.
    threshold : float
        Максимально допустимый VIF.
    verbose : bool
        Выводить ли ход фильтрации.

    Returns
    -------
    selected : list[str]
        Список оставшихся признаков.
    final_vif : pd.DataFrame
        Значения VIF после фильтрации.
    history : pd.DataFrame
        История исключения признаков.
    """
    if threshold < 1:
        raise ValueError("Порог VIF должен быть >= 1.")

    selected = list(X.columns)
    history = []

    iteration = 0

    while len(selected) > 1:
        current_vif = calculate_vif(X[selected])

        max_row = current_vif.iloc[0]
        max_vif = float(max_row["VIF"])

        if max_vif <= threshold:
            break

        feature_to_remove = str(max_row["feature"])
        iteration += 1

        history.append({
            "iteration": iteration,
            "removed_feature": feature_to_remove,
            "VIF_before_removal": max_vif,
            "features_remaining": len(selected) - 1
        })

        selected.remove(feature_to_remove)

        if verbose and (
            iteration <= 10 or iteration % 10 == 0
        ):
            print(
                f"Итерация {iteration}: удалён "
                f"{feature_to_remove}, VIF={max_vif:.2f}"
            )

    final_vif = calculate_vif(X[selected])

    return (
        selected,
        final_vif,
        pd.DataFrame(
            history,
            columns=[
                "iteration",
                "removed_feature",
                "VIF_before_removal",
                "features_remaining"
            ]
        )
    )
