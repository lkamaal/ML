import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from forecasting_wrapper import ForecastingWrapper


# ============================================================
# ФИКСТУРЫ
# ============================================================

@pytest.fixture(scope="module")
def synthetic_series():

    rng = np.random.default_rng(42)

    n = 180

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    t = np.arange(n)

    values = (
        8
        + 0.01 * t
        + 10 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 1.0, n)
    )

    return pd.Series(
        values,
        index=dates,
        name="temperature_2m_mean"
    )


@pytest.fixture(scope="module")
def fitted_arima(synthetic_series):

    model = ForecastingWrapper(
        model_type="ARIMA",
        order=(2, 0, 1)
    )

    return model.fit(synthetic_series)


# ============================================================
# ТЕСТ 1. СОЗДАНИЕ МОДЕЛИ
# ============================================================

def test_model_initialization():

    model = ForecastingWrapper(
        model_type="ARIMA",
        order=(2, 0, 1)
    )

    assert model.model_type == "ARIMA"
    assert model.is_fitted_ is False


# ============================================================
# ТЕСТ 2. НЕИЗВЕСТНАЯ МОДЕЛЬ
# ============================================================

def test_invalid_model_type():

    with pytest.raises(ValueError):

        ForecastingWrapper(
            model_type="UNKNOWN"
        )


# ============================================================
# ТЕСТ 3. ОБУЧЕНИЕ ТРЁХ МОДЕЛЕЙ
# ============================================================

@pytest.mark.parametrize(
    "model_type",
    ["ARIMA", "SARIMA", "SARIMAX"]
)
def test_fit_supported_models(
    synthetic_series,
    model_type
):

    model = ForecastingWrapper(
        model_type=model_type,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7)
    )

    model.fit(synthetic_series)

    assert model.is_fitted_
    assert model.result_ is not None


# ============================================================
# ТЕСТ 4. ПРОГНОЗ НА 30 ДНЕЙ
# ============================================================

def test_forecast_length(fitted_arima):

    forecast = fitted_arima.forecast(
        steps=30
    )

    assert len(forecast) == 30


# ============================================================
# ТЕСТ 5. КОРРЕКТНОСТЬ ДАТ
# ============================================================

def test_forecast_dates(
    fitted_arima,
    synthetic_series
):

    forecast = fitted_arima.forecast(
        steps=7
    )

    expected = pd.date_range(
        synthetic_series.index[-1]
        + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    assert list(forecast["date"]) == list(expected)


# ============================================================
# ТЕСТ 6. КОНЕЧНОСТЬ ПРОГНОЗОВ
# ============================================================

def test_forecast_finite(fitted_arima):

    forecast = fitted_arima.forecast(
        steps=30
    )

    assert np.isfinite(
        forecast[
            ["forecast", "lower", "upper"]
        ].to_numpy()
    ).all()


# ============================================================
# ТЕСТ 7. КОРРЕКТНОСТЬ ИНТЕРВАЛОВ
# ============================================================

def test_prediction_intervals(fitted_arima):

    forecast = fitted_arima.forecast(
        steps=30
    )

    assert (
        forecast["lower"]
        <= forecast["upper"]
    ).all()

    assert (
        forecast["forecast"]
        >= forecast["lower"]
    ).all()

    assert (
        forecast["forecast"]
        <= forecast["upper"]
    ).all()


# ============================================================
# ТЕСТ 8. МЕТОД PREDICT
# ============================================================

def test_predict_date_range(
    fitted_arima,
    synthetic_series
):

    start = (
        synthetic_series.index[-1]
        + pd.Timedelta(days=1)
    )

    end = start + pd.Timedelta(days=6)

    prediction = fitted_arima.predict(
        start=start,
        end=end
    )

    assert len(prediction) == 7
    assert prediction["date"].iloc[0] == start
    assert prediction["date"].iloc[-1] == end


# ============================================================
# ТЕСТ 9. ОТКАЗ ДО ОБУЧЕНИЯ
# ============================================================

def test_forecast_before_fit():

    model = ForecastingWrapper()

    with pytest.raises(RuntimeError):
        model.forecast(steps=7)


# ============================================================
# ТЕСТ 10. НЕПРАВИЛЬНЫЙ ГОРИЗОНТ
# ============================================================

@pytest.mark.parametrize(
    "steps",
    [0, -1, -10]
)
def test_invalid_forecast_steps(
    fitted_arima,
    steps
):

    with pytest.raises(ValueError):
        fitted_arima.forecast(steps=steps)


# ============================================================
# ТЕСТ 11. НЕПРАВИЛЬНЫЙ ALPHA
# ============================================================

@pytest.mark.parametrize(
    "alpha",
    [-0.1, 0, 1, 1.5]
)
def test_invalid_alpha(
    fitted_arima,
    alpha
):

    with pytest.raises(ValueError):

        fitted_arima.forecast(
            steps=7,
            alpha=alpha
        )


# ============================================================
# ТЕСТ 12. ПРОПУСКИ
# ============================================================

def test_missing_values(synthetic_series):

    broken = synthetic_series.copy()

    broken.iloc[10] = np.nan

    model = ForecastingWrapper()

    with pytest.raises(ValueError):
        model.fit(broken)


# ============================================================
# ТЕСТ 13. ДУБЛИКАТЫ ДАТ
# ============================================================

def test_duplicate_dates(synthetic_series):

    broken = synthetic_series.copy()

    dates = list(broken.index)

    dates[5] = dates[4]

    broken.index = pd.DatetimeIndex(dates)

    model = ForecastingWrapper()

    with pytest.raises(ValueError):
        model.fit(broken)


# ============================================================
# ТЕСТ 14. НЕПРАВИЛЬНЫЙ ТИП ИНДЕКСА
# ============================================================

def test_invalid_index(synthetic_series):

    broken = synthetic_series.copy()

    broken.index = range(len(broken))

    model = ForecastingWrapper()

    with pytest.raises(TypeError):
        model.fit(broken)


# ============================================================
# ТЕСТ 15. МАЛО НАБЛЮДЕНИЙ
# ============================================================

def test_insufficient_observations(
    synthetic_series
):

    short_series = synthetic_series.iloc[:20]

    model = ForecastingWrapper()

    with pytest.raises(ValueError):
        model.fit(short_series)


# ============================================================
# ТЕСТ 16. ДИАГНОСТИКА
# ============================================================

def test_diagnostics(fitted_arima):

    result = fitted_arima.diagnose(
        lags=(7, 14),
        burn=30
    )

    assert result["n_residuals"] > 0
    assert np.isfinite(result["mean_residual"])
    assert np.isfinite(result["std_residual"])

    assert 7 in result["ljung_box"]
    assert 14 in result["ljung_box"]

    assert 0 <= result["shapiro"]["p_value"] <= 1
    assert (
        0
        <= result["breusch_pagan"]["lm_p_value"]
        <= 1
    )


# ============================================================
# ТЕСТ 17. СОХРАНЕНИЕ И ЗАГРУЗКА
# ============================================================

def test_save_load(
    fitted_arima,
    tmp_path
):

    path = tmp_path / "model.joblib"

    original = fitted_arima.forecast(
        steps=30
    )

    fitted_arima.save(path)

    restored = ForecastingWrapper.load(path)

    recovered = restored.forecast(
        steps=30
    )

    assert path.exists()

    np.testing.assert_allclose(
        original["forecast"].to_numpy(),
        recovered["forecast"].to_numpy(),
        rtol=0,
        atol=1e-10
    )

    np.testing.assert_allclose(
        original[
            ["lower", "upper"]
        ].to_numpy(),
        recovered[
            ["lower", "upper"]
        ].to_numpy(),
        rtol=0,
        atol=1e-10
    )


# ============================================================
# ТЕСТ 18. НЕЗАВИСИМОСТЬ ОТ БУДУЩИХ ДАННЫХ
# ============================================================

def test_future_target_independence(
    fitted_arima,
    synthetic_series
):

    forecast_original = fitted_arima.forecast(
        steps=7
    )

    # Изменяем будущие фактические значения.
    # Они не передаются модели при прогнозе.

    rng = np.random.default_rng(123)

    future_actual = pd.Series(
        rng.normal(0, 100, 7),
        index=pd.date_range(
            synthetic_series.index[-1]
            + pd.Timedelta(days=1),
            periods=7,
            freq="D"
        )
    )

    assert len(future_actual) == 7

    forecast_again = fitted_arima.forecast(
        steps=7
    )

    np.testing.assert_allclose(
        forecast_original["forecast"].to_numpy(),
        forecast_again["forecast"].to_numpy(),
        rtol=0,
        atol=1e-10
    )


# ============================================================
# ТЕСТ 19. КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# ============================================================

def test_model_config_and_params(fitted_arima):

    config = fitted_arima.get_config()

    params = fitted_arima.get_params()

    assert config["model_type"] == "ARIMA"
    assert config["is_fitted"] is True

    assert not params.empty

    assert {
        "parameter",
        "coefficient",
        "p_value"
    }.issubset(params.columns)


# ============================================================
# ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: НЕПРЕРЫВНОСТЬ ДАТ
# ============================================================

def test_missing_calendar_day(synthetic_series):

    broken = synthetic_series.drop(
        synthetic_series.index[20]
    )

    model = ForecastingWrapper()

    with pytest.raises(ValueError):
        model.fit(broken)


# ============================================================
# ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: PREDICT В ПРОШЛОЕ
# ============================================================

def test_predict_in_sample_rejected(
    fitted_arima,
    synthetic_series
):

    with pytest.raises(ValueError):

        fitted_arima.predict(
            start=synthetic_series.index[-10],
            end=synthetic_series.index[-1]
        )
