
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

TARGET = "temperature_2m_mean"
LAGS = [1, 2, 3, 7, 14, 30]
WINDOWS = [3, 7, 14, 30]


def load_dataset(path):
    df = pd.read_csv(path)
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_convert("Europe/Moscow")
    )
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.normalize()
    return df


def build_safe_features(df):
    dates = pd.to_datetime(df["date"])
    y = df[TARGET].astype(float)

    X = pd.DataFrame(index=df.index)

    X["month_sin"] = np.sin(
        2 * np.pi * dates.dt.month / 12
    )
    X["month_cos"] = np.cos(
        2 * np.pi * dates.dt.month / 12
    )
    X["dayofyear_sin"] = np.sin(
        2 * np.pi * dates.dt.dayofyear / 365.25
    )
    X["dayofyear_cos"] = np.cos(
        2 * np.pi * dates.dt.dayofyear / 365.25
    )
    X["dayofweek"] = dates.dt.dayofweek

    for lag in LAGS:
        X[f"lag_{lag}"] = y.shift(lag)

    past = y.shift(1)

    for window in WINDOWS:
        X[f"rolling_mean_{window}"] = (
            past.rolling(window, min_periods=window).mean()
        )
        X[f"rolling_std_{window}"] = (
            past.rolling(window, min_periods=window).std()
        )

    return X


class TimeSeriesStructureTransformer(
    BaseEstimator, TransformerMixin
):
    def __init__(
        self,
        n_clusters=3,
        random_state=42,
        n_init=10
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        array = self._validate(X)

        self.n_features_in_ = array.shape[1]
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(array)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.kmeans_.fit(scaled)

        return self

    def transform(self, X):
        check_is_fitted(self, ["scaler_", "kmeans_"])

        array = self._validate(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError("Неверное число признаков")

        scaled = self.scaler_.transform(array)
        labels = self.kmeans_.predict(scaled)
        distances = self.kmeans_.transform(scaled)

        one_hot = np.column_stack([
            (labels == i).astype(float)
            for i in range(self.n_clusters)
        ])

        return np.column_stack([
            array, one_hot, distances
        ])

    @staticmethod
    def _validate(X):
        array = np.asarray(X, dtype=float)

        if array.ndim != 2:
            raise ValueError("Ожидается двумерный массив")

        if not np.isfinite(array).all():
            raise ValueError("NaN или бесконечность")

        return array
