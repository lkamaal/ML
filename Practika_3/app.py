from pathlib import Path
import io
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from forecasting_wrapper import ForecastingWrapper

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
MODEL_DIR = PROJECT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEFAULT = "temperature_2m_mean"

st.title("📈 Time Series Forecasting")
st.caption(
    "Классические модели прогнозирования временных рядов: "
    "ARIMA, SARIMA и SARIMAX"
)

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

st.sidebar.header("1. Источник данных")

source = st.sidebar.radio(
    "Выберите источник",
    ["Данные проекта", "Загрузить CSV"]
)

if source == "Данные проекта":

    candidates = [
        DATA_DIR / "task1_prepared_series.csv",
        PROJECT.parent / "Practika_1"
        / "data" / "features_dataset.csv",
        PROJECT.parent / "Practika_1"
        / "data" / "processed_dataset.csv",
        PROJECT.parent / "Practika_2"
        / "data" / "features_dataset.csv"
    ]

    path = next(
        (p for p in candidates if p.is_file()),
        None
    )

    if path is None:
        st.error("Не найден исходный файл проекта.")
        st.stop()

    df = pd.read_csv(path)

    st.sidebar.success(
        f"Загружен файл: {path.name}"
    )

else:

    uploaded = st.sidebar.file_uploader(
        "Выберите CSV",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Загрузите CSV-файл для начала работы.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Ошибка чтения CSV: {exc}")
        st.stop()

# ============================================================
# 3. ВЫБОР КОЛОНОК
# ============================================================

st.sidebar.header("2. Структура данных")

columns = list(df.columns)

date_candidates = [
    c for c in ["date", "timestamp", "time"]
    if c in columns
]

date_default = (
    columns.index(date_candidates[0])
    if date_candidates else 0
)

date_col = st.sidebar.selectbox(
    "Столбец даты",
    columns,
    index=date_default
)

target_default = (
    columns.index(TARGET_DEFAULT)
    if TARGET_DEFAULT in columns
    else 0
)

target_col = st.sidebar.selectbox(
    "Целевая переменная",
    columns,
    index=target_default
)

if date_col == target_col:
    st.error(
        "Столбец даты и целевая переменная "
        "не должны совпадать."
    )
    st.stop()

# ============================================================
# 4. ПОДГОТОВКА РЯДА
# ============================================================

prepared = df[[date_col, target_col]].copy()

prepared[date_col] = pd.to_datetime(
    prepared[date_col],
    errors="coerce",
    utc=True
)

prepared[target_col] = pd.to_numeric(
    prepared[target_col],
    errors="coerce"
)

missing_dates = int(
    prepared[date_col].isna().sum()
)

missing_values = int(
    prepared[target_col].isna().sum()
)

if missing_dates or missing_values:
    st.error(
        f"Пропуски: даты — {missing_dates}, "
        f"значения — {missing_values}."
    )
    st.stop()

prepared[date_col] = (
    prepared[date_col].dt.tz_convert(None)
)

prepared = prepared.sort_values(
    date_col
)

duplicates = int(
    prepared[date_col].duplicated().sum()
)

if duplicates:
    st.error(
        f"Обнаружено повторяющихся дат: {duplicates}"
    )
    st.stop()

series = pd.Series(
    prepared[target_col].to_numpy(dtype=float),
    index=pd.DatetimeIndex(
        prepared[date_col]
    ),
    name=target_col
).asfreq("D")

if series.isna().any():
    st.error(
        "Во временном ряду отсутствуют некоторые дни. "
        "Необходимо подготовить непрерывный дневной ряд."
    )
    st.stop()

if len(series) < 100:
    st.error(
        "Для работы приложения требуется "
        "не менее 100 наблюдений."
    )
    st.stop()

# ============================================================
# 5. ОБЩАЯ ИНФОРМАЦИЯ
# ============================================================

st.subheader("Исходные данные")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Наблюдений",
    f"{len(series):,}"
)

c2.metric(
    "Начало",
    str(series.index.min().date())
)

c3.metric(
    "Конец",
    str(series.index.max().date())
)

c4.metric(
    "Среднее значение",
    f"{series.mean():.2f}"
)

with st.expander("Просмотр данных"):

    st.dataframe(
        series.reset_index().rename(
            columns={
                "index": "date",
                target_col: "value"
            }
        ),
        use_container_width=True
    )

# ============================================================
# 6. ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

st.sidebar.header("3. Параметры модели")

model_type = st.sidebar.selectbox(
    "Модель",
    ["ARIMA", "SARIMA", "SARIMAX"]
)

strategy = st.sidebar.selectbox(
    "Стратегия",
    ["Recursive"]
)

st.sidebar.caption(
    "Для классических моделей реализована "
    "рекурсивная стратегия многошагового прогноза. "
    "Direct и Hybrid требуют отдельной реализации."
)

horizon = st.sidebar.slider(
    "Горизонт прогнозирования",
    min_value=1,
    max_value=90,
    value=30
)

p = st.sidebar.number_input(
    "p — порядок AR",
    min_value=0,
    max_value=10,
    value=5 if model_type == "ARIMA" else 2
)

d = st.sidebar.number_input(
    "d — порядок дифференцирования",
    min_value=0,
    max_value=2,
    value=0
)

q = st.sidebar.number_input(
    "q — порядок MA",
    min_value=0,
    max_value=10,
    value=2 if model_type == "ARIMA" else 1
)

seasonal_order = (0, 0, 0, 0)

if model_type in ["SARIMA", "SARIMAX"]:

    st.sidebar.subheader("Сезонные параметры")

    P = st.sidebar.number_input(
        "P",
        min_value=0,
        max_value=3,
        value=1
    )

    D = st.sidebar.number_input(
        "D",
        min_value=0,
        max_value=2,
        value=0
    )

    Q = st.sidebar.number_input(
        "Q",
        min_value=0,
        max_value=3,
        value=1
    )

    s = st.sidebar.number_input(
        "Период сезонности",
        min_value=2,
        max_value=365,
        value=7
    )

    seasonal_order = (
        int(P), int(D), int(Q), int(s)
    )

# ============================================================
# 7. РЕЖИМ ОЦЕНКИ
# ============================================================

st.sidebar.header("4. Режим работы")

mode = st.sidebar.radio(
    "Режим",
    [
        "Прогноз будущих значений",
        "Оценка на тестовой выборке"
    ]
)

train_share = 0.80

if mode == "Оценка на тестовой выборке":

    train_share = st.sidebar.slider(
        "Доля обучающей выборки",
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05
    )

    split = int(len(series) * train_share)

    train = series.iloc[:split]
    test = series.iloc[split:]

    if len(test) < horizon:
        st.error(
            "Тестовая выборка короче горизонта прогноза."
        )
        st.stop()

else:

    train = series.copy()
    test = None

# ============================================================
# 8. ОБУЧЕНИЕ И ПРОГНОЗИРОВАНИЕ
# ============================================================

st.subheader("Обучение и прогнозирование")

run = st.button(
    "Обучить модель и построить прогноз",
    type="primary"
)

if run:

    config = {
        "model_type": model_type,
        "order": (int(p), int(d), int(q))
    }

    if model_type in ["SARIMA", "SARIMAX"]:
        config["seasonal_order"] = seasonal_order

    try:

        with st.spinner("Обучение модели..."):

            model = ForecastingWrapper(**config)

            model.fit(train)

            forecast_df = model.forecast(
                steps=horizon,
                alpha=0.05
            )

            diagnostics = model.diagnose()

        st.session_state["model"] = model
        st.session_state["forecast"] = forecast_df
        st.session_state["diagnostics"] = diagnostics
        st.session_state["train"] = train
        st.session_state["test"] = test
        st.session_state["config"] = config
        st.session_state["mode"] = mode

        st.success("Модель успешно обучена.")

    except Exception as exc:

        st.error(
            f"Ошибка: {type(exc).__name__}: {exc}"
        )

# ============================================================
# 9. РЕЗУЛЬТАТЫ
# ============================================================

if "model" in st.session_state:

    model = st.session_state["model"]
    forecast_df = st.session_state["forecast"]
    diagnostics = st.session_state["diagnostics"]

    train_result = st.session_state["train"]
    test_result = st.session_state["test"]

    st.subheader("График прогнозирования")

    fig = go.Figure()

    history = train_result.iloc[-180:]

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            name="Исторические данные",
            mode="lines"
        )
    )

    if test_result is not None:

        actual = test_result.iloc[
            :len(forecast_df)
        ]

        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                name="Фактические значения",
                mode="lines"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            name="Прогноз",
            mode="lines",
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(width=0),
            name="95% прогнозный интервал"
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Дата",
        yaxis_title="Значение",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # 10. МЕТРИКИ
    # ========================================================

    if test_result is not None:

        actual = test_result.iloc[
            :len(forecast_df)
        ].to_numpy(dtype=float)

        predicted = (
            forecast_df["forecast"]
            .to_numpy(dtype=float)
        )

        errors = actual - predicted

        mae = float(
            np.mean(np.abs(errors))
        )

        rmse = float(
            np.sqrt(np.mean(errors ** 2))
        )

        bias = float(
            np.mean(predicted - actual)
        )

        st.subheader("Метрики качества")

        m1, m2, m3 = st.columns(3)

        m1.metric("MAE", f"{mae:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("Bias", f"{bias:.4f}")

    # ========================================================
    # 11. ДИАГНОСТИКА
    # ========================================================

    st.subheader("Диагностика остатков")

    d1, d2, d3 = st.columns(3)

    lb_p = diagnostics[
        "ljung_box"
    ][30]["p_value"]

    shapiro_p = diagnostics[
        "shapiro"
    ]["p_value"]

    bp_p = diagnostics[
        "breusch_pagan"
    ]["lm_p_value"]

    d1.metric(
        "Ljung–Box p-value",
        f"{lb_p:.6f}"
    )

    d2.metric(
        "Shapiro p-value",
        f"{shapiro_p:.6f}"
    )

    d3.metric(
        "Breusch–Pagan p-value",
        f"{bp_p:.6f}"
    )

    with st.expander("Коэффициенты модели"):

        st.dataframe(
            model.get_params(),
            use_container_width=True
        )

    # ========================================================
    # 12. ТАБЛИЦА ПРОГНОЗОВ
    # ========================================================

    st.subheader("Таблица прогнозов")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # ========================================================
    # 13. ВЫГРУЗКА ПРОГНОЗОВ
    # ========================================================

    csv_bytes = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Скачать прогноз CSV",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv"
    )

    # ========================================================
    # 14. JSON-ОТЧЁТ
    # ========================================================

    report = {
        "model": model.model_type,
        "config": model.get_config(),
        "forecast_horizon": len(forecast_df),
        "train_observations": len(train_result),
        "diagnostics": diagnostics
    }

    if test_result is not None:
        report["metrics"] = {
            "MAE": mae,
            "RMSE": rmse,
            "bias": bias
        }

    report_json = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    st.download_button(
        "Скачать JSON-отчёт",
        data=report_json,
        file_name="forecast_report.json",
        mime="application/json"
    )

    # ========================================================
    # 15. СОХРАНЕНИЕ МОДЕЛИ
    # ========================================================

    st.subheader("Сохранение модели")

    save_name = st.text_input(
        "Имя файла модели",
        value=f"{model.model_type.lower()}_streamlit.joblib"
    )

    if st.button("Сохранить модель"):

        safe_name = Path(save_name).name

        if not safe_name.endswith(".joblib"):
            safe_name += ".joblib"

        save_path = MODEL_DIR / safe_name

        model.save(save_path)

        st.success(
            f"Модель сохранена: {save_path}"
        )

st.divider()

st.caption(
    "Практическая работа №3. "
    "Классическое прогнозирование временных рядов."
)
