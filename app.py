
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. НАСТРОЙКА ПРИЛОЖЕНИЯ
# ============================================================

st.set_page_config(
    page_title="Анализ температуры Москвы",
    page_icon="🌡️",
    layout="wide"
)

TARGET = "temperature_2m_mean"
DATE_COLUMN = "time"

DATA_PATH = Path("data/features_dataset.csv")
FEATURE_PATH = Path("reports/task10_feature_list.csv")


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN].astype(str).str[:10],
        format="%Y-%m-%d"
    )

    df = df.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    return df


@st.cache_data
def load_features():

    features = pd.read_csv(FEATURE_PATH)

    return features["feature"].tolist()


df = load_data()
feature_columns = load_features()

missing = [
    col for col in feature_columns
    if col not in df.columns
]

if missing:
    st.error(f"Не найдены признаки: {missing}")
    st.stop()


# ============================================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

@st.cache_resource
def train_model():

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy()

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

    model.fit(
        train_df[feature_columns],
        train_df[TARGET]
    )

    test_df["ridge_prediction"] = model.predict(
        test_df[feature_columns]
    )

    lag1 = f"{TARGET}_lag_1"

    test_df["naive_prediction"] = test_df[lag1]

    return model, test_df


model, test_df = train_model()


# ============================================================
# 4. БОКОВАЯ ПАНЕЛЬ
# ============================================================

st.sidebar.title("🌡️ Анализ температуры")

page = st.sidebar.radio(
    "Выберите раздел",
    [
        "Обзор данных",
        "Визуальный анализ",
        "Прогнозирование",
        "Информация о модели"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Практическая работа №1"
)

st.sidebar.write(
    "Предобработка и инженерия признаков"
)

st.sidebar.write(
    "Город: Москва"
)


# ============================================================
# 5. ОБЗОР ДАННЫХ
# ============================================================

if page == "Обзор данных":

    st.title("Обзор датасета")

    st.write(
        "Данные о среднесуточной температуре "
        "воздуха в Москве."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Наблюдений",
        len(df)
    )

    col2.metric(
        "Признаков модели",
        len(feature_columns)
    )

    col3.metric(
        "Средняя температура",
        f"{df[TARGET].mean():.2f} °C"
    )

    col4.metric(
        "Пропусков в target",
        int(df[TARGET].isna().sum())
    )

    st.subheader("Исходная таблица")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "Скачать датасет CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="moscow_weather_features.csv",
        mime="text/csv"
    )


# ============================================================
# 6. ВИЗУАЛЬНЫЙ АНАЛИЗ
# ============================================================

elif page == "Визуальный анализ":

    st.title("Визуальный анализ температуры")

    min_date = df[DATE_COLUMN].min().date()
    max_date = df[DATE_COLUMN].max().date()

    date_range = st.date_input(
        "Выберите период",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:

        start_date, end_date = date_range

        filtered = df[
            (df[DATE_COLUMN].dt.date >= start_date)
            &
            (df[DATE_COLUMN].dt.date <= end_date)
        ].copy()

    else:
        filtered = df.copy()

    st.subheader("Динамика температуры")

    fig = px.line(
        filtered,
        x=DATE_COLUMN,
        y=TARGET,
        title="Среднесуточная температура Москвы",
        labels={
            DATE_COLUMN: "Дата",
            TARGET: "Температура, °C"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Распределение температуры")

    fig_hist = px.histogram(
        filtered,
        x=TARGET,
        nbins=40,
        title="Распределение среднесуточной температуры"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.subheader("Средняя температура по месяцам")

    monthly = filtered.copy()

    monthly["month"] = monthly[DATE_COLUMN].dt.month

    monthly_mean = (
        monthly.groupby("month")[TARGET]
        .mean()
        .reset_index()
    )

    fig_month = px.bar(
        monthly_mean,
        x="month",
        y=TARGET,
        title="Средняя температура по месяцам"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

    st.subheader("Корреляции признаков")

    selected_features = st.multiselect(
        "Выберите признаки",
        options=feature_columns,
        default=feature_columns[:8]
    )

    if selected_features:

        corr_columns = selected_features + [TARGET]

        corr_matrix = df[
            list(dict.fromkeys(corr_columns))
        ].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Корреляционная матрица"
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )


# ============================================================
# 7. ПРОГНОЗИРОВАНИЕ
# ============================================================

elif page == "Прогнозирование":

    st.title("Прогнозирование температуры")

    st.write(
        "Сравнение Ridge-регрессии и наивного "
        "прогноза на тестовой выборке."
    )

    ridge_mae = mean_absolute_error(
        test_df[TARGET],
        test_df["ridge_prediction"]
    )

    ridge_rmse = np.sqrt(
        mean_squared_error(
            test_df[TARGET],
            test_df["ridge_prediction"]
        )
    )

    ridge_r2 = r2_score(
        test_df[TARGET],
        test_df["ridge_prediction"]
    )

    naive_mae = mean_absolute_error(
        test_df[TARGET],
        test_df["naive_prediction"]
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            test_df[TARGET],
            test_df["naive_prediction"]
        )
    )

    naive_r2 = r2_score(
        test_df[TARGET],
        test_df["naive_prediction"]
    )

    metrics = pd.DataFrame([
        {
            "Модель": "Ridge",
            "MAE": ridge_mae,
            "RMSE": ridge_rmse,
            "R2": ridge_r2
        },
        {
            "Модель": "Наивный прогноз",
            "MAE": naive_mae,
            "RMSE": naive_rmse,
            "R2": naive_r2
        }
    ])

    st.subheader("Метрики качества")

    st.dataframe(
        metrics.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Фактические и прогнозируемые значения")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_df[DATE_COLUMN],
            y=test_df[TARGET],
            mode="lines",
            name="Фактическая температура"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_df[DATE_COLUMN],
            y=test_df["ridge_prediction"],
            mode="lines",
            name="Ridge"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_df[DATE_COLUMN],
            y=test_df["naive_prediction"],
            mode="lines",
            name="Наивный прогноз"
        )
    )

    fig.update_layout(
        title="Сравнение прогнозов",
        xaxis_title="Дата",
        yaxis_title="Температура, °C"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Прогноз для выбранной даты")

    selected_date = st.selectbox(
        "Выберите дату из тестовой выборки",
        options=test_df[DATE_COLUMN].dt.strftime(
            "%Y-%m-%d"
        ).tolist()
    )

    selected_row = test_df[
        test_df[DATE_COLUMN].dt.strftime(
            "%Y-%m-%d"
        ) == selected_date
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Фактическая температура",
        f"{selected_row[TARGET]:.2f} °C"
    )

    col2.metric(
        "Прогноз Ridge",
        f"{selected_row['ridge_prediction']:.2f} °C"
    )

    col3.metric(
        "Наивный прогноз",
        f"{selected_row['naive_prediction']:.2f} °C"
    )

    st.download_button(
        "Скачать результаты прогнозирования",
        data=test_df[
            [
                DATE_COLUMN,
                TARGET,
                "ridge_prediction",
                "naive_prediction"
            ]
        ].to_csv(index=False).encode("utf-8-sig"),
        file_name="temperature_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# 8. ИНФОРМАЦИЯ О МОДЕЛИ
# ============================================================

elif page == "Информация о модели":

    st.title("Информация о модели")

    st.subheader("Описание")

    st.write(
        "Модель прогнозирования среднесуточной "
        "температуры воздуха Москвы."
    )

    st.write("Алгоритм: Ridge Regression")
    st.write(f"Количество признаков: {len(feature_columns)}")
    st.write("Коэффициент регуляризации: alpha = 10")

    st.subheader("Этапы предобработки")

    st.code(
        "SimpleImputer -> StandardScaler -> Ridge",
        language="text"
    )

    st.subheader("Разделение данных")

    st.write("Train: 70%")
    st.write("Validation: 15%")
    st.write("Test: 15%")

    st.subheader("Используемые признаки")

    st.dataframe(
        pd.DataFrame({
            "Признак": feature_columns
        }),
        use_container_width=True
    )

    st.info(
        "Приложение демонстрирует прогнозирование "
        "на исторической тестовой выборке. "
        "Прогноз будущих дат без подготовленных "
        "признаков не реализован."
    )
