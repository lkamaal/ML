from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from scipy.stats import shapiro
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ForecastingWrapper:
    """
    Универсальный интерфейс ARIMA, SARIMA, SARIMAX.

    Методы:
        fit()
        predict()
        forecast()
        diagnose()
        save()
        load()

    Данные:
        pandas.Series с DatetimeIndex и дневной частотой.

    ВАЖНО:
        predict() выполняет прогноз за пределами
        обучающей выборки, а не in-sample prediction.
    """

    SUPPORTED_MODELS = {
        "ARIMA",
        "SARIMA",
        "SARIMAX"
    }

    def __init__(
        self,
        model_type="ARIMA",
        order=(5, 0, 2),
        seasonal_order=(1, 0, 1, 7),
        trend="c"
    ):

        model_type = str(model_type).upper()

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Неизвестная модель: {model_type}"
            )

        self.model_type = model_type
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend

        self.model_ = None
        self.result_ = None
        self.train_series_ = None
        self.is_fitted_ = False

    @staticmethod
    def _validate_series(series):

        if not isinstance(series, pd.Series):
            raise TypeError(
                "Ожидается pandas.Series."
            )

        if not isinstance(
            series.index,
            pd.DatetimeIndex
        ):
            raise TypeError(
                "Индекс должен быть DatetimeIndex."
            )

        series = series.sort_index().copy()

        if series.index.has_duplicates:
            raise ValueError(
                "Обнаружены повторяющиеся даты."
            )

        if series.isna().any():
            raise ValueError(
                "Обнаружены пропущенные значения."
            )

        if not np.issubdtype(
            series.dtype,
            np.number
        ):
            raise TypeError(
                "Значения ряда должны быть числовыми."
            )

        if not np.isfinite(
            series.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Ряд содержит бесконечные значения."
            )

        if len(series) < 50:
            raise ValueError(
                "Недостаточно наблюдений для обучения."
            )

        expected_index = pd.date_range(
            start=series.index[0],
            periods=len(series),
            freq="D"
        )

        if not series.index.equals(expected_index):
            raise ValueError(
                "Временной ряд должен иметь "
                "непрерывную дневную частоту."
            )

        return series.asfreq("D")

    @staticmethod
    def _calendar_features(index):

        index = pd.DatetimeIndex(index)

        day = index.dayofyear.to_numpy()

        return pd.DataFrame(
            {
                "annual_sin_1": np.sin(
                    2 * np.pi * day / 365.25
                ),
                "annual_cos_1": np.cos(
                    2 * np.pi * day / 365.25
                ),
                "annual_sin_2": np.sin(
                    4 * np.pi * day / 365.25
                ),
                "annual_cos_2": np.cos(
                    4 * np.pi * day / 365.25
                )
            },
            index=index
        )

    def fit(self, series):

        series = self._validate_series(series)

        self.train_series_ = series

        if self.model_type == "ARIMA":

            self.model_ = ARIMA(
                series,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.result_ = self.model_.fit()

        elif self.model_type == "SARIMA":

            self.model_ = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.result_ = self.model_.fit(
                disp=False,
                maxiter=100
            )

        elif self.model_type == "SARIMAX":

            exog = self._calendar_features(
                series.index
            )

            self.model_ = SARIMAX(
                series,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.result_ = self.model_.fit(
                disp=False,
                maxiter=100
            )

        self.is_fitted_ = True

        return self

    def _check_fitted(self):

        if not self.is_fitted_:
            raise RuntimeError(
                "Сначала вызовите fit()."
            )

    def forecast(
        self,
        steps=30,
        alpha=0.05
    ):

        self._check_fitted()

        steps = int(steps)

        if steps <= 0:
            raise ValueError(
                "steps должен быть положительным."
            )

        if not 0 < alpha < 1:
            raise ValueError(
                "alpha должен быть между 0 и 1."
            )

        last_date = self.train_series_.index[-1]

        future_index = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=steps,
            freq="D"
        )

        if self.model_type == "SARIMAX":

            future_exog = self._calendar_features(
                future_index
            )

            forecast_result = (
                self.result_.get_forecast(
                    steps=steps,
                    exog=future_exog
                )
            )

        else:

            forecast_result = (
                self.result_.get_forecast(
                    steps=steps
                )
            )

        mean = np.asarray(
            forecast_result.predicted_mean,
            dtype=float
        )

        intervals = np.asarray(
            forecast_result.conf_int(alpha=alpha),
            dtype=float
        )

        return pd.DataFrame(
            {
                "date": future_index,
                "forecast": mean,
                "lower": intervals[:, 0],
                "upper": intervals[:, 1]
            }
        )

    def predict(
        self,
        start=None,
        end=None
    ):

        """
        Возвращает out-of-sample прогноз
        для указанного интервала дат.

        start/end должны находиться строго
        после конца обучающей выборки.
        """

        self._check_fitted()

        last_date = self.train_series_.index[-1]

        if start is None:
            start = (
                last_date + pd.Timedelta(days=1)
            )
        else:
            start = pd.Timestamp(start)

        if end is None:
            end = start
        else:
            end = pd.Timestamp(end)

        if start <= last_date:
            raise ValueError(
                "predict() поддерживает только "
                "даты после окончания обучения."
            )

        if end < start:
            raise ValueError(
                "end не может быть раньше start."
            )

        steps = int(
            (end - last_date).days
        )

        result = self.forecast(
            steps=steps
        )

        return result[
            (result["date"] >= start)
            & (result["date"] <= end)
        ].reset_index(drop=True)

    def diagnose(
        self,
        lags=(7, 14, 21, 30),
        burn=30
    ):

        self._check_fitted()

        residuals = np.asarray(
            self.result_.resid,
            dtype=float
        )

        residuals = residuals[
            int(burn):
        ]

        residuals = residuals[
            np.isfinite(residuals)
        ]

        if len(residuals) < 40:
            raise ValueError(
                "Недостаточно остатков для диагностики."
            )

        # Тест Льюнга — Бокса.
        lb = acorr_ljungbox(
            residuals,
            lags=list(lags),
            return_df=True
        )

        # Тест Шапиро — Уилка.
        shapiro_stat, shapiro_p = shapiro(
            residuals
        )

        # Тест Бройша — Пагана.
        n = len(residuals)
        t = np.arange(n)

        exog_bp = np.column_stack([
            np.ones(n),
            (t - t.mean()) / max(t.std(), 1),
            np.sin(2 * np.pi * t / 365.25),
            np.cos(2 * np.pi * t / 365.25)
        ])

        bp = het_breuschpagan(
            residuals,
            exog_bp
        )

        return {
            "model": self.model_type,
            "n_residuals": int(n),
            "mean_residual": float(
                np.mean(residuals)
            ),
            "std_residual": float(
                np.std(residuals, ddof=1)
            ),
            "ljung_box": {
                int(lag): {
                    "statistic": float(
                        lb.loc[lag, "lb_stat"]
                    ),
                    "p_value": float(
                        lb.loc[lag, "lb_pvalue"]
                    )
                }
                for lag in lags
            },
            "shapiro": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p)
            },
            "breusch_pagan": {
                "lm_statistic": float(bp[0]),
                "lm_p_value": float(bp[1]),
                "f_statistic": float(bp[2]),
                "f_p_value": float(bp[3])
            }
        }

    def save(self, path):

        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self,
            path
        )

        return path

    @classmethod
    def load(cls, path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        loaded = joblib.load(path)

        if not isinstance(loaded, cls):
            raise TypeError(
                "Файл не содержит ForecastingWrapper."
            )

        loaded._check_fitted()

        return loaded

    def get_params(self):

        self._check_fitted()

        return pd.DataFrame({
            "parameter": self.result_.param_names,
            "coefficient": np.asarray(
                self.result_.params
            ),
            "p_value": np.asarray(
                self.result_.pvalues
            )
        })

    def get_config(self):

        return {
            "model_type": self.model_type,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "is_fitted": self.is_fitted_
        }
