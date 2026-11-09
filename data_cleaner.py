
"""
Модуль первичной очистки табличных данных.

Функциональность:
- унификация временных меток;
- удаление полных дубликатов;
- проверка временной оси;
- обработка выбросов методом IQR;
- передискретизация временных рядов;
- настройка операций через YAML.
"""

import pandas as pd
import numpy as np
import yaml


class DataCleaner:
    """
    Класс для первичной очистки табличных данных.

    Parameters
    ----------
    config_path : str
        Путь к YAML-файлу с настройками очистки.
    """

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.config = config["cleaning"]
        self.date_column = self.config["date_column"]
        self.timezone = self.config["timezone"]

        self.report = {}

    def convert_datetime(self, df):
        """
        Приводит временные метки к единому часовому поясу.

        Наивные временные метки интерпретируются как
        локальное время, указанное в конфигурации.
        """

        df = df.copy()

        dates = pd.to_datetime(
            df[self.date_column],
            errors="coerce"
        )

        invalid_dates = dates.isna().sum()

        if invalid_dates > 0:
            raise ValueError(
                f"Обнаружено некорректных дат: {invalid_dates}"
            )

        if dates.dt.tz is None:
            dates = dates.dt.tz_localize(
                self.timezone,
                ambiguous="raise",
                nonexistent="raise"
            )
        else:
            dates = dates.dt.tz_convert(self.timezone)

        df[self.date_column] = dates

        self.report["invalid_dates"] = int(invalid_dates)

        return df

    def remove_duplicates(self, df):
        """
        Удаляет полные дубликаты строк.
        """

        df = df.copy()

        duplicates = df.duplicated().sum()

        df = df.drop_duplicates().reset_index(drop=True)

        self.report["duplicates_removed"] = int(duplicates)

        return df

    def check_monotonicity(self, df):
        """
        Проверяет хронологический порядок временных меток.

        При нарушении порядка сортирует строки по дате.
        """

        df = df.copy()

        dates = df[self.date_column]

        is_monotonic = dates.is_monotonic_increasing

        self.report["monotonic_before"] = bool(is_monotonic)

        if not is_monotonic:
            df = df.sort_values(
                self.date_column
            ).reset_index(drop=True)

        self.report["monotonic_after"] = bool(
            df[self.date_column].is_monotonic_increasing
        )

        return df

    def handle_outliers(self, df):
        """
        Обрабатывает выбросы методом IQR.

        Значения ниже Q1 - 1.5 * IQR и выше
        Q3 + 1.5 * IQR ограничиваются
        соответствующими границами.

        Примечание: границы рассчитываются по
        переданному датасету. Для ML-пайплайна
        их следует оценивать только на train.
        """

        df = df.copy()

        settings = self.config["outliers"]

        multiplier = settings["multiplier"]
        columns = settings["columns"]
        action = settings["action"]

        outlier_report = {}

        for column in columns:

            if column not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR

            mask = (
                (df[column] < lower) |
                (df[column] > upper)
            )

            outlier_report[column] = int(mask.sum())

            if action == "clip":
                df[column] = df[column].clip(
                    lower=lower,
                    upper=upper
                )
            elif action != "none":
                raise ValueError(
                    f"Неизвестное действие с выбросами: {action}"
                )

        self.report["outliers_detected"] = outlier_report

        return df

    def resample_data(self, df):
        """
        Передискретизирует временной ряд.

        Поддерживаемые стратегии:
        mean, sum, last.
        """

        settings = self.config["resampling"]

        if not settings["enabled"]:
            self.report["resampling"] = "disabled"
            return df

        df = df.copy()

        frequency = settings["frequency"]
        aggregation = settings["aggregation"]

        if aggregation not in ["mean", "sum", "last"]:
            raise ValueError(
                "Агрегация должна быть mean, sum или last"
            )

        df = df.set_index(self.date_column)

        if aggregation == "mean":
            df = df.resample(frequency).mean(numeric_only=True)

        elif aggregation == "sum":
            df = df.resample(frequency).sum(
                numeric_only=True,
                min_count=1
            )

        elif aggregation == "last":
            df = df.resample(frequency).last()

        df = df.reset_index()

        self.report["resampling"] = frequency

        return df

    def clean(self, df):
        """
        Выполняет очистку в порядке из конфигурации.

        Parameters
        ----------
        df : pandas.DataFrame
            Исходный датасет.

        Returns
        -------
        pandas.DataFrame
            Очищенный датасет.
        """

        self.report = {}
        df = df.copy()

        operations = {
            "datetime": self.convert_datetime,
            "duplicates": self.remove_duplicates,
            "monotonicity": self.check_monotonicity,
            "outliers": self.handle_outliers,
            "resampling": self.resample_data
        }

        order = self.config["operations"]

        if len(order) != len(set(order)):
            raise ValueError("Операции не должны повторяться")

        for operation in order:

            if operation not in operations:
                raise ValueError(
                    f"Неизвестная операция: {operation}"
                )

        # Защищаем обязательный порядок зависимых операций.
        dependencies = [
            ("datetime", "monotonicity"),
            ("duplicates", "outliers"),
            ("monotonicity", "resampling")
        ]

        for first, second in dependencies:
            if first in order and second in order:
                if order.index(first) > order.index(second):
                    raise ValueError(
                        f"Операция {first} должна идти перед {second}"
                    )

        for operation in order:
            df = operations[operation](df)

        self.report["final_rows"] = len(df)

        return df

    def get_report(self):
        """
        Возвращает отчёт о выполненной очистке.
        """
        return self.report.copy()
