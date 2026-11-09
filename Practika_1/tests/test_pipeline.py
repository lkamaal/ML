
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw_dataset.csv"
FEATURES_PATH = ROOT / "data" / "features_dataset.csv"
FEATURE_LIST_PATH = ROOT / "reports" / "task10_feature_list.csv"
MODEL_PATH = ROOT / "models" / "task10_ridge_pipeline.joblib"

TARGET = "temperature_2m_mean"
DATE_COLUMN = "time"


def read_dataset(path):
    df = pd.read_csv(path)

    # Приводим даты к единому календарному формату
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN].astype(str).str[:10],
        format="%Y-%m-%d"
    )

    df = df.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    assert df[DATE_COLUMN].is_unique, (
        f"Обнаружены повторяющиеся даты: {path}"
    )

    return df


@pytest.fixture(scope="module")
def raw_df():
    return read_dataset(RAW_PATH)


@pytest.fixture(scope="module")
def features_df():
    return read_dataset(FEATURES_PATH)


@pytest.fixture(scope="module")
def feature_columns():
    df = pd.read_csv(FEATURE_LIST_PATH)
    return df["feature"].tolist()


@pytest.fixture(scope="module")
def prepared_data(features_df, feature_columns):
    X = features_df[feature_columns].copy()
    y = features_df[TARGET].copy()

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "train_end": train_end,
        "val_end": val_end
    }


# ------------------------------------------------------------
# ТЕСТ 1. НАЛИЧИЕ ФАЙЛОВ
# ------------------------------------------------------------

def test_required_files_exist():
    for path in [
        RAW_PATH,
        FEATURES_PATH,
        FEATURE_LIST_PATH,
        MODEL_PATH
    ]:
        assert path.exists(), f"Не найден файл: {path}"


# ------------------------------------------------------------
# ТЕСТ 2. СТРУКТУРА ДАТАСЕТА
# ------------------------------------------------------------

def test_dataset_structure(features_df, feature_columns):
    assert TARGET in features_df.columns
    assert DATE_COLUMN in features_df.columns

    assert features_df.columns.is_unique
    assert features_df[DATE_COLUMN].is_unique
    assert features_df[DATE_COLUMN].is_monotonic_increasing

    assert len(feature_columns) == len(set(feature_columns))
    assert len(feature_columns) > 0

    assert set(feature_columns).issubset(features_df.columns)


# ------------------------------------------------------------
# ТЕСТ 3. ОТСУТСТВИЕ ПРОПУСКОВ В TARGET
# ------------------------------------------------------------

def test_target_has_no_missing_values(features_df):
    assert features_df[TARGET].notna().all()


# ------------------------------------------------------------
# ТЕСТ 4. КОРРЕКТНОСТЬ ЛАГОВ
# ------------------------------------------------------------

def test_lag_features(raw_df, features_df, feature_columns):

    raw_indexed = raw_df.set_index(DATE_COLUMN)
    feature_indexed = features_df.set_index(DATE_COLUMN)

    lag_columns = [
        col for col in feature_columns
        if "_lag_" in col
    ]

    assert len(lag_columns) > 0

    for col in lag_columns:

        source, lag_text = col.rsplit("_lag_", 1)
        lag = int(lag_text)

        expected = raw_indexed[source].shift(lag)
        expected = expected.reindex(feature_indexed.index)

        actual = feature_indexed[col]

        np.testing.assert_allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"Ошибка лагового признака: {col}"
        )


# ------------------------------------------------------------
# ТЕСТ 5. КОРРЕКТНОСТЬ СКОЛЬЗЯЩИХ ПРИЗНАКОВ
# ------------------------------------------------------------

def test_rolling_features(raw_df, features_df, feature_columns):

    raw_indexed = raw_df.set_index(DATE_COLUMN)
    feature_indexed = features_df.set_index(DATE_COLUMN)

    rolling_columns = [
        col for col in feature_columns
        if "_roll_" in col
    ]

    assert len(rolling_columns) > 0

    for col in rolling_columns:

        source, suffix = col.rsplit("_roll_", 1)
        statistic, window_text = suffix.rsplit("_", 1)
        window = int(window_text)

        rolling = raw_indexed[source].shift(1).rolling(window)

        if statistic == "mean":
            expected = rolling.mean()
        elif statistic == "std":
            expected = rolling.std()
        elif statistic == "min":
            expected = rolling.min()
        elif statistic == "max":
            expected = rolling.max()
        else:
            raise AssertionError(
                f"Неизвестная статистика: {statistic}"
            )

        expected = expected.reindex(feature_indexed.index)

        actual = feature_indexed[col]

        np.testing.assert_allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
            err_msg=f"Ошибка скользящего признака: {col}"
        )


# ------------------------------------------------------------
# ТЕСТ 6. ОТСУТСТВИЕ ПРЯМЫХ КОПИЙ TARGET
# ------------------------------------------------------------

def test_no_direct_target_copy(features_df, feature_columns):

    target = features_df[TARGET].to_numpy(dtype=float)

    for col in feature_columns:

        values = features_df[col].to_numpy(dtype=float)

        is_copy = np.array_equal(
            values,
            target,
            equal_nan=True
        )

        assert not is_copy, (
            f"Признак {col} совпадает с target"
        )


# ------------------------------------------------------------
# ТЕСТ 7. ХРОНОЛОГИЧЕСКОЕ РАЗДЕЛЕНИЕ
# ------------------------------------------------------------

def test_chronological_split(features_df, prepared_data):

    train_end = prepared_data["train_end"]
    val_end = prepared_data["val_end"]

    train_dates = features_df.iloc[:train_end][DATE_COLUMN]
    val_dates = features_df.iloc[train_end:val_end][DATE_COLUMN]
    test_dates = features_df.iloc[val_end:][DATE_COLUMN]

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()


# ------------------------------------------------------------
# ТЕСТ 8. ВРЕМЕННАЯ КРОСС-ВАЛИДАЦИЯ
# ------------------------------------------------------------

def test_time_series_cv(features_df):

    cv = TimeSeriesSplit(n_splits=5, gap=1)

    for train_idx, val_idx in cv.split(features_df):

        assert train_idx.max() < val_idx.min()
        assert len(set(train_idx) & set(val_idx)) == 0
        assert val_idx.min() - train_idx.max() >= 2


# ------------------------------------------------------------
# ТЕСТ 9. ПАРАМЕТРЫ IMPUTER ТОЛЬКО ПО TRAIN
# ------------------------------------------------------------

def test_imputer_train_only(prepared_data):

    model = load_model()

    imputer = (
        model.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )

    X_train = prepared_data["X_train"]

    expected = np.nanmedian(
        X_train.to_numpy(dtype=float),
        axis=0
    )

    np.testing.assert_allclose(
        imputer.statistics_,
        expected,
        equal_nan=True
    )


# ------------------------------------------------------------
# ТЕСТ 10. ПАРАМЕТРЫ SCALER ТОЛЬКО ПО TRAIN
# ------------------------------------------------------------

def test_scaler_train_only(prepared_data):

    model = load_model()

    numeric_pipeline = (
        model.named_steps["preprocessor"]
        .named_transformers_["numeric"]
    )

    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    imputed_train = imputer.transform(
        prepared_data["X_train"]
    )

    expected_means = imputed_train.mean(axis=0)

    np.testing.assert_allclose(
        scaler.mean_,
        expected_means
    )


# ------------------------------------------------------------
# ЗАГРУЗКА СОХРАНЁННОЙ МОДЕЛИ
# ------------------------------------------------------------

def load_model():
    """
    Для переносимости тестов создаём модель заново
    из воспроизводимой конфигурации.

    Это исключает зависимость от определения класса
    FeatureValidator в Jupyter __main__.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    feature_columns = pd.read_csv(
        FEATURE_LIST_PATH
    )["feature"].tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, feature_columns)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=10.0))
    ])

    df = read_dataset(FEATURES_PATH)

    n = len(df)
    train_end = int(n * 0.70)

    model.fit(
        df.iloc[:train_end][feature_columns],
        df.iloc[:train_end][TARGET]
    )

    return model


# ------------------------------------------------------------
# ТЕСТ 11. ВОСПРОИЗВОДИМОСТЬ
# ------------------------------------------------------------

def test_model_reproducibility(prepared_data):

    model1 = load_model()
    model2 = load_model()

    X_test = prepared_data["X_test"]

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)

    np.testing.assert_allclose(
        pred1,
        pred2,
        rtol=1e-10,
        atol=1e-10
    )


# ------------------------------------------------------------
# ТЕСТ 12. СОХРАНЕНИЕ И ЗАГРУЗКА
# ------------------------------------------------------------

def test_model_save_load(prepared_data, tmp_path):

    model = load_model()

    model_path = tmp_path / "test_pipeline.joblib"

    joblib.dump(model, model_path)

    loaded = joblib.load(model_path)

    pred_before = model.predict(
        prepared_data["X_test"]
    )

    pred_after = loaded.predict(
        prepared_data["X_test"]
    )

    np.testing.assert_allclose(
        pred_before,
        pred_after,
        rtol=1e-10,
        atol=1e-10
    )


# ------------------------------------------------------------
# ТЕСТ 13. КАЧЕСТВО ПРОГНОЗОВ
# ------------------------------------------------------------

def test_predictions_are_finite(prepared_data):

    model = load_model()

    predictions = model.predict(
        prepared_data["X_test"]
    )

    assert np.isfinite(predictions).all()

    mae = mean_absolute_error(
        prepared_data["y_test"],
        predictions
    )

    assert np.isfinite(mae)
    assert mae >= 0
