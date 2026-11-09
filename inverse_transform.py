
import numpy as np
import pandas as pd

from scipy.stats import boxcox
from scipy.special import inv_boxcox


class ReversibleTransformer:
    """
    Класс для прямого и обратного преобразования временного ряда.

    Поддерживаемые методы:
    - log: логарифмирование со сдвигом;
    - boxcox: преобразование Бокса — Кокса;
    - diff: первая разность;
    - seasonal_diff: сезонная разность;
    - combined_diff: комбинированная разность.

    Исходные граничные значения сохраняются для
    последующего точного обратного восстановления.
    """

    def __init__(self, method, seasonal_period=365):
        self.method = method
        self.seasonal_period = seasonal_period
        self.shift = 0.0
        self.lmbda = None
        self.initial_values = None
        self.original_index = None
        self.original_name = None

    def fit_transform(self, series):
        """
        Выполняет преобразование и сохраняет параметры.

        Parameters
        ----------
        series : pandas.Series
            Исходный временной ряд без пропусков.

        Returns
        -------
        pandas.Series
            Преобразованный временной ряд.
        """

        series = series.astype(float).copy()

        if series.empty or series.isna().any():
            raise ValueError("Ряд пуст или содержит пропуски")

        self.original_index = series.index.copy()
        self.original_name = series.name

        if self.method == "log":

            self.shift = max(0.0, 1.0 - series.min())

            result = np.log(series + self.shift)

        elif self.method == "boxcox":

            self.shift = max(0.0, 1.0 - series.min())

            values, self.lmbda = boxcox(
                series.values + self.shift
            )

            result = pd.Series(
                values,
                index=series.index,
                name=series.name
            )

        elif self.method == "diff":

            self.initial_values = series.iloc[:1].copy()

            result = series.diff().dropna()

        elif self.method == "seasonal_diff":

            s = self.seasonal_period

            if len(series) <= s:
                raise ValueError("Недостаточно наблюдений")

            self.initial_values = series.iloc[:s].copy()

            result = series.diff(s).dropna()

        elif self.method == "combined_diff":

            s = self.seasonal_period

            if len(series) <= s + 1:
                raise ValueError("Недостаточно наблюдений")

            self.initial_values = series.iloc[:s + 1].copy()

            result = (
                series.diff(s)
                .diff()
                .dropna()
            )

        else:
            raise ValueError(
                f"Неизвестный метод: {self.method}"
            )

        return result

    def inverse_transform(self, transformed):
        """
        Восстанавливает исходный ряд из преобразованного.

        Для разностей используются сохранённые начальные
        значения исходного ряда.
        """

        transformed = transformed.astype(float).copy()

        if self.original_index is None:
            raise ValueError("Сначала выполните fit_transform")

        if self.method == "log":

            restored = np.exp(transformed) - self.shift

            return pd.Series(
                restored,
                index=transformed.index,
                name=self.original_name
            )

        elif self.method == "boxcox":

            restored = (
                inv_boxcox(transformed.values, self.lmbda)
                - self.shift
            )

            return pd.Series(
                restored,
                index=transformed.index,
                name=self.original_name
            )

        elif self.method in [
            "diff", "seasonal_diff", "combined_diff"
        ]:

            s = self.seasonal_period

            if self.method == "diff":
                initial_count = 1

            elif self.method == "seasonal_diff":
                initial_count = s

            else:
                initial_count = s + 1

            expected_length = (
                len(self.original_index) - initial_count
            )

            if len(transformed) != expected_length:
                raise ValueError(
                    "Длина преобразованного ряда не совпадает "
                    "с ожидаемой длиной"
                )

            values = np.empty(len(self.original_index))

            values[:initial_count] = (
                self.initial_values.values
            )

            diff_values = transformed.to_numpy()

            for i in range(initial_count, len(values)):

                j = i - initial_count

                if self.method == "diff":

                    values[i] = (
                        diff_values[j] + values[i - 1]
                    )

                elif self.method == "seasonal_diff":

                    values[i] = (
                        diff_values[j] + values[i - s]
                    )

                elif self.method == "combined_diff":

                    values[i] = (
                        diff_values[j]
                        + values[i - 1]
                        + values[i - s]
                        - values[i - s - 1]
                    )

            return pd.Series(
                values,
                index=self.original_index,
                name=self.original_name
            )
