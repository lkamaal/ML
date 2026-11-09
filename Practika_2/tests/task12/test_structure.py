
import numpy as np
import pandas as pd
import pytest
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

from structure_module import (
    TARGET,
    load_dataset,
    build_safe_features,
    TimeSeriesStructureTransformer
)


@pytest.fixture(scope="module")
def dataset(data_path):
    return load_dataset(data_path)


@pytest.fixture(scope="module")
def prepared(dataset):
    features = build_safe_features(dataset)
    valid = features.notna().all(axis=1)

    X = features.loc[valid].reset_index(drop=True)
    y = dataset.loc[valid, TARGET].reset_index(drop=True)

    split = int(len(X) * 0.8)

    return (
        X.iloc[:split].copy(),
        X.iloc[split:].copy(),
        y.iloc[:split].copy(),
        y.iloc[split:].copy()
    )


def test_dataset_nonempty(dataset):
    assert len(dataset) > 0


def test_dataset_no_missing_target(dataset):
    assert dataset[TARGET].notna().all()


def test_dataset_unique_dates(dataset):
    assert dataset["date"].is_unique


def test_dataset_chronological(dataset):
    assert dataset["date"].is_monotonic_increasing


def test_dataset_daily_frequency(dataset):
    differences = dataset["date"].diff().dropna()
    assert differences.eq(pd.Timedelta(days=1)).all()


def test_lag_1(dataset):
    X = build_safe_features(dataset)
    expected = dataset[TARGET].shift(1)

    np.testing.assert_allclose(
        X["lag_1"],
        expected,
        equal_nan=True
    )


def test_rolling_7(dataset):
    X = build_safe_features(dataset)

    expected = (
        dataset[TARGET]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    np.testing.assert_allclose(
        X["rolling_mean_7"],
        expected,
        equal_nan=True
    )


def test_future_values_do_not_change_past(dataset):
    original = build_safe_features(dataset)

    modified = dataset.copy()
    cutoff = len(dataset) // 2

    modified.loc[
        cutoff + 1:, TARGET
    ] += 1000

    changed = build_safe_features(modified)

    np.testing.assert_allclose(
        original.iloc[:cutoff + 1],
        changed.iloc[:cutoff + 1],
        equal_nan=True
    )


def test_features_have_no_nan_after_warmup(prepared):
    X_train, X_test, _, _ = prepared

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_test.to_numpy()).all()


def test_transformer_fit_returns_self(prepared):
    X_train, _, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    assert model.fit(X_train) is model


def test_transformer_output_shape(prepared):
    X_train, X_test, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)

    result = model.transform(X_test)

    assert result.shape == (
        len(X_test),
        X_test.shape[1] + 6
    )


def test_fit_transform_equivalence(prepared):
    X_train, _, _, _ = prepared

    model1 = TimeSeriesStructureTransformer()
    model2 = TimeSeriesStructureTransformer()

    first = model1.fit_transform(X_train)
    second = model2.fit(X_train).transform(X_train)

    np.testing.assert_allclose(first, second)


def test_transform_does_not_refit(prepared):
    X_train, X_test, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)

    centers = model.kmeans_.cluster_centers_.copy()
    means = model.scaler_.mean_.copy()

    model.transform(X_test)

    np.testing.assert_array_equal(
        centers, model.kmeans_.cluster_centers_
    )
    np.testing.assert_array_equal(
        means, model.scaler_.mean_
    )


def test_transform_is_deterministic(prepared):
    X_train, X_test, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)

    first = model.transform(X_test)
    second = model.transform(X_test)

    np.testing.assert_allclose(first, second)


def test_input_not_modified(prepared):
    X_train, X_test, _, _ = prepared

    original = X_test.copy(deep=True)

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)
    model.transform(X_test)

    pd.testing.assert_frame_equal(X_test, original)


def test_clone_compatibility():
    model = TimeSeriesStructureTransformer(
        n_clusters=4, random_state=42
    )

    cloned = clone(model)

    assert cloned.get_params() == model.get_params()


def test_serialization(prepared, tmp_path):
    X_train, X_test, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)

    path = tmp_path / "transformer.joblib"
    joblib.dump(model, path)

    restored = joblib.load(path)

    np.testing.assert_allclose(
        model.transform(X_test),
        restored.transform(X_test)
    )


def test_pipeline_fit_predict(prepared):
    X_train, X_test, y_train, _ = prepared

    pipeline = Pipeline([
        (
            "structure",
            TimeSeriesStructureTransformer()
        ),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    assert len(prediction) == len(X_test)
    assert np.isfinite(prediction).all()


def test_pipeline_serialization(prepared, tmp_path):
    X_train, X_test, y_train, _ = prepared

    pipeline = Pipeline([
        (
            "structure",
            TimeSeriesStructureTransformer()
        ),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train, y_train)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)

    restored = joblib.load(path)

    np.testing.assert_allclose(
        pipeline.predict(X_test),
        restored.predict(X_test)
    )


def test_time_series_split_order(prepared):
    X_train, X_test, _, _ = prepared
    X = pd.concat([X_train, X_test], ignore_index=True)

    cv = TimeSeriesSplit(n_splits=5)

    for train_idx, test_idx in cv.split(X):
        assert train_idx.max() < test_idx.min()
        assert not set(train_idx).intersection(test_idx)


def test_random_target_no_strong_signal(dataset):
    # Несколько независимых случайных рядов.
    # Проверяем, что модель не получает устойчиво
    # высокий R² при отсутствии реального сигнала.

    scores = []

    for seed in range(5):
        rng = np.random.default_rng(seed)

        random_data = dataset.copy()
        random_data[TARGET] = rng.normal(
            size=len(random_data)
        )

        X_all = build_safe_features(random_data)
        valid = X_all.notna().all(axis=1)

        X = X_all.loc[valid].reset_index(drop=True)
        y = random_data.loc[
            valid, TARGET
        ].reset_index(drop=True)

        split = int(len(X) * 0.8)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        model.fit(X.iloc[:split], y.iloc[:split])

        predictions = model.predict(X.iloc[split:])

        scores.append(
            r2_score(y.iloc[split:], predictions)
        )

    # Это диагностический порог, а не доказательство
    # отсутствия любых утечек.
    assert np.mean(scores) < 0.2


def test_invalid_nan_rejected(prepared):
    X_train, _, _, _ = prepared

    model = TimeSeriesStructureTransformer()

    bad = X_train.copy()
    bad.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):
        model.fit(bad)


def test_invalid_feature_count_rejected(prepared):
    X_train, X_test, _, _ = prepared

    model = TimeSeriesStructureTransformer()
    model.fit(X_train)

    with pytest.raises(ValueError):
        model.transform(X_test.iloc[:, :-1])
