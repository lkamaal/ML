
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"

DATE_COL = "date"
TARGET = "temperature_2m_mean"


@pytest.fixture(scope="module")
def datasets():

    raw = pd.read_csv(
        DATA_DIR / "task1_dataset.csv",
        parse_dates=["time"]
    )

    # Исходные временные метки соответствуют
    # 21:00 UTC предыдущего календарного дня.
    raw = raw.rename(columns={"time": DATE_COL})

    raw[DATE_COL] = (
        raw[DATE_COL] + pd.Timedelta(hours=3)
    ).dt.normalize()

    trainval = pd.read_csv(
        DATA_DIR / "task3_trainval.csv",
        parse_dates=[DATE_COL]
    )

    test = pd.read_csv(
        DATA_DIR / "task3_test.csv",
        parse_dates=[DATE_COL]
    )

    raw = raw.sort_values(DATE_COL).reset_index(drop=True)
    trainval = trainval.sort_values(DATE_COL).reset_index(drop=True)
    test = test.sort_values(DATE_COL).reset_index(drop=True)

    return raw, trainval, test


@pytest.fixture(scope="module")
def modeling_data(datasets):

    _, trainval, _ = datasets

    features = [
        c for c in trainval.columns
        if c not in [DATE_COL, TARGET]
    ]

    X = trainval[features].astype(float)
    y = trainval[TARGET].astype(float)

    return X, y


# ============================================================
# БЛОК 1. ПРОВЕРКИ ДАННЫХ
# ============================================================

def test_01_files_exist():

    files = [
        "task1_dataset.csv",
        "task3_trainval.csv",
        "task3_test.csv"
    ]

    for filename in files:
        assert (DATA_DIR / filename).exists()


def test_02_datasets_not_empty(datasets):

    for frame in datasets:
        assert len(frame) > 0


def test_03_target_exists(datasets):

    for frame in datasets:
        assert TARGET in frame.columns


def test_04_dates_are_unique(datasets):

    for frame in datasets:
        assert frame[DATE_COL].is_unique


def test_05_dates_are_sorted(datasets):

    for frame in datasets:
        assert frame[DATE_COL].is_monotonic_increasing


def test_06_daily_frequency(datasets):

    for frame in datasets:

        differences = (
            frame[DATE_COL]
            .diff()
            .dropna()
        )

        assert (
            differences == pd.Timedelta(days=1)
        ).all()


def test_07_no_missing_modeling_values(datasets):

    _, trainval, test = datasets

    assert not trainval.isna().any().any()
    assert not test.isna().any().any()


def test_08_train_test_do_not_overlap(datasets):

    _, trainval, test = datasets

    train_dates = set(trainval[DATE_COL])
    test_dates = set(test[DATE_COL])

    assert train_dates.isdisjoint(test_dates)

    assert (
        trainval[DATE_COL].max()
        < test[DATE_COL].min()
    )


# ============================================================
# БЛОК 2. ПРОВЕРКИ ПРИЗНАКОВ
# ============================================================

def test_09_targets_match_raw(datasets):

    raw, trainval, test = datasets

    raw_target = raw.set_index(DATE_COL)[TARGET]

    for frame in [trainval, test]:

        expected = raw_target.reindex(
            frame[DATE_COL]
        )

        assert expected.notna().all()

        np.testing.assert_allclose(
            frame[TARGET].to_numpy(),
            expected.to_numpy(),
            rtol=0,
            atol=1e-8
        )


@pytest.mark.parametrize(
    "lag",
    [1, 2, 3, 7, 14, 21, 30, 60, 90]
)
def test_10_lag_features(datasets, lag):

    raw, trainval, _ = datasets

    column = f"target_lag_{lag}"

    assert column in trainval.columns

    series = raw.set_index(DATE_COL)[TARGET]

    expected_dates = (
        trainval[DATE_COL]
        - pd.Timedelta(days=lag)
    )

    expected = series.reindex(
        expected_dates
    )

    assert expected.notna().all()

    np.testing.assert_allclose(
        trainval[column].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-8
    )


@pytest.mark.parametrize(
    "window",
    [3, 7, 14, 30]
)
def test_11_rolling_mean(datasets, window):

    raw, trainval, _ = datasets

    column = f"target_roll_mean_{window}"

    series = raw.set_index(DATE_COL)[TARGET]

    expected = (
        series.shift(1)
        .rolling(window)
        .mean()
        .reindex(trainval[DATE_COL])
    )

    assert expected.notna().all()

    np.testing.assert_allclose(
        trainval[column].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-8
    )


@pytest.mark.parametrize(
    "window",
    [3, 7, 14, 30]
)
def test_12_rolling_std(datasets, window):

    raw, trainval, _ = datasets

    column = f"target_roll_std_{window}"

    series = raw.set_index(DATE_COL)[TARGET]

    expected = (
        series.shift(1)
        .rolling(window)
        .std(ddof=1)
        .reindex(trainval[DATE_COL])
    )

    assert expected.notna().all()

    np.testing.assert_allclose(
        trainval[column].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-8
    )


@pytest.mark.parametrize(
    "window",
    [3, 7, 14, 30]
)
def test_13_rolling_min(datasets, window):

    raw, trainval, _ = datasets

    column = f"target_roll_min_{window}"

    series = raw.set_index(DATE_COL)[TARGET]

    expected = (
        series.shift(1)
        .rolling(window)
        .min()
        .reindex(trainval[DATE_COL])
    )

    assert expected.notna().all()

    np.testing.assert_allclose(
        trainval[column].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-8
    )


@pytest.mark.parametrize(
    "window",
    [3, 7, 14, 30]
)
def test_14_rolling_max(datasets, window):

    raw, trainval, _ = datasets

    column = f"target_roll_max_{window}"

    series = raw.set_index(DATE_COL)[TARGET]

    expected = (
        series.shift(1)
        .rolling(window)
        .max()
        .reindex(trainval[DATE_COL])
    )

    assert expected.notna().all()

    np.testing.assert_allclose(
        trainval[column].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-8
    )


# ============================================================
# БЛОК 3. ВРЕМЕННАЯ КРОСС-ВАЛИДАЦИЯ
# ============================================================

def test_15_timeseries_split_is_chronological(
    modeling_data
):

    X, _ = modeling_data

    splitter = TimeSeriesSplit(n_splits=5)

    for train_idx, val_idx in splitter.split(X):

        assert train_idx.max() < val_idx.min()

        assert set(train_idx).isdisjoint(
            set(val_idx)
        )


# ============================================================
# БЛОК 4. МОДЕЛИ
# ============================================================

def test_16_model_fit_predict(modeling_data):

    X, y = modeling_data

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    model.fit(X.iloc[:350], y.iloc[:350])

    predictions = model.predict(
        X.iloc[350:400]
    )

    assert len(predictions) == 50
    assert np.isfinite(predictions).all()


def test_17_predictions_have_correct_shape(
    modeling_data
):

    X, y = modeling_data

    model = Ridge(alpha=1.0)

    model.fit(
        X.iloc[:300],
        y.iloc[:300]
    )

    predictions = model.predict(
        X.iloc[300:320]
    )

    assert predictions.shape == (20,)


def test_18_metrics_are_valid(modeling_data):

    X, y = modeling_data

    model = Ridge(alpha=1.0)

    model.fit(
        X.iloc[:350],
        y.iloc[:350]
    )

    predictions = model.predict(
        X.iloc[350:400]
    )

    actual = y.iloc[350:400]

    mae = mean_absolute_error(
        actual,
        predictions
    )

    assert np.isfinite(mae)
    assert mae >= 0


def test_19_residuals_are_correct(modeling_data):

    X, y = modeling_data

    model = Ridge(alpha=1.0)

    model.fit(
        X.iloc[:350],
        y.iloc[:350]
    )

    predictions = model.predict(
        X.iloc[350:400]
    )

    residuals = (
        y.iloc[350:400].to_numpy()
        - predictions
    )

    assert len(residuals) == 50
    assert np.isfinite(residuals).all()

    np.testing.assert_allclose(
        residuals + predictions,
        y.iloc[350:400].to_numpy()
    )


def test_20_joblib_roundtrip(
    modeling_data,
    tmp_path
):

    X, y = modeling_data

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    model.fit(
        X.iloc[:350],
        y.iloc[:350]
    )

    before = model.predict(
        X.iloc[350:400]
    )

    path = tmp_path / "test_model.joblib"

    joblib.dump(model, path)

    restored = joblib.load(path)

    after = restored.predict(
        X.iloc[350:400]
    )

    np.testing.assert_allclose(
        before,
        after,
        rtol=0,
        atol=1e-10
    )
