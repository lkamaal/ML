
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"

TARGET = "temperature_2m_mean"

st.set_page_config(
    page_title="Temperature Structure Explorer",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Temperature Structure Explorer")
st.caption(
    "Структурный анализ среднесуточной температуры Москвы, "
    "2024–2025 годы"
)

@st.cache_data
def load_main():
    path = DATA_DIR / "features_dataset.csv"
    df = pd.read_csv(path)

    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_convert("Europe/Moscow")
    )
    df["date"] = (
        df["time"].dt.tz_localize(None).dt.normalize()
    )

    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def load_optional(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def normalize_dates(data, columns):
    result = data.copy()
    for col in columns:
        if col in result.columns:
            result[col] = (
                pd.to_datetime(
                    result[col], utc=True, errors="coerce"
                )
                .dt.tz_localize(None)
                .dt.normalize()
            )
    return result

def temperature_chart(data, title):
    fig = px.line(
        data,
        x="date",
        y=TARGET,
        title=title,
        labels={
            "date": "Дата",
            TARGET: "Температура, °C"
        }
    )
    fig.update_layout(
        hovermode="x unified",
        height=450
    )
    return fig

try:
    df = load_main()
except Exception as exc:
    st.error(f"Не удалось загрузить исходные данные: {exc}")
    st.stop()

# ------------------------------------------------------------
# Боковая панель
# ------------------------------------------------------------

st.sidebar.header("Параметры анализа")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Период исследования",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) != 2:
    st.info("Выберите начальную и конечную даты.")
    st.stop()

start_date, end_date = selected_dates

filtered = df[
    df["date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date)
    )
].copy()

if filtered.empty:
    st.warning("За выбранный период нет наблюдений.")
    st.stop()

section = st.sidebar.radio(
    "Раздел",
    [
        "Обзор",
        "Аномалии",
        "Кластеризация",
        "Сегментация"
    ]
)

# ------------------------------------------------------------
# 1. ОБЗОР
# ------------------------------------------------------------

if section == "Обзор":

    st.header("Обзор временного ряда")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Наблюдений", len(filtered))
    c2.metric(
        "Средняя температура",
        f"{filtered[TARGET].mean():.2f} °C"
    )
    c3.metric(
        "Минимум",
        f"{filtered[TARGET].min():.2f} °C"
    )
    c4.metric(
        "Максимум",
        f"{filtered[TARGET].max():.2f} °C"
    )

    st.plotly_chart(
        temperature_chart(
            filtered, "Динамика температуры"
        ),
        use_container_width=True
    )

    st.subheader("Распределение температуры")

    fig = px.histogram(
        filtered,
        x=TARGET,
        nbins=35,
        marginal="box",
        labels={
            TARGET: "Температура, °C"
        }
    )
    st.plotly_chart(
        fig, use_container_width=True
    )

    st.subheader("Статистические характеристики")

    statistics = filtered[TARGET].describe()
    st.dataframe(
        statistics.to_frame("Значение"),
        use_container_width=True
    )

    st.subheader("Средняя температура по месяцам")

    monthly = filtered.copy()
    monthly["month"] = (
        monthly["date"].dt.to_period("M").astype(str)
    )

    monthly_stats = (
        monthly.groupby("month", as_index=False)[TARGET]
        .mean()
    )

    fig = px.bar(
        monthly_stats,
        x="month",
        y=TARGET,
        labels={
            "month": "Месяц",
            TARGET: "Средняя температура, °C"
        }
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

# ------------------------------------------------------------
# 2. АНОМАЛИИ
# ------------------------------------------------------------

elif section == "Аномалии":

    st.header("Аномальные наблюдения")

    anomalies = load_optional(
        DATA_DIR / "task2_anomalies_dataset.csv"
    )

    if anomalies is None:
        st.warning(
            "Не найден файл task2_anomalies_dataset.csv"
        )
        st.stop()

    # В исходном файле могут присутствовать
    # различные названия столбцов аномалий.
    candidate_columns = [
        col for col in anomalies.columns
        if (
            "anomal" in col.lower()
            or "outlier" in col.lower()
        )
    ]

    st.write("Доступные столбцы аномалий:")
    st.write(candidate_columns)

    if not candidate_columns:
        st.warning(
            "В файле не обнаружены столбцы с "
            "названиями anomaly/outlier."
        )
        st.dataframe(anomalies.head())
        st.stop()

    selected_col = st.selectbox(
        "Выберите метод обнаружения аномалий",
        candidate_columns
    )

    # Выравниваем результаты по датам, а не
    # по порядковым индексам после фильтрации.
    if "time" in anomalies.columns:
        anomalies["date"] = (
            pd.to_datetime(
                anomalies["time"],
                utc=True,
                errors="coerce"
            )
            .dt.tz_convert("Europe/Moscow")
            .dt.tz_localize(None)
            .dt.normalize()
        )
    elif "date" in anomalies.columns:
        anomalies = normalize_dates(
            anomalies, ["date"]
        )
    else:
        st.error(
            "В файле аномалий нет столбца date/time."
        )
        st.stop()

    merged = filtered.merge(
        anomalies[["date", selected_col]],
        on="date",
        how="left",
        validate="one_to_one"
    )

    values = merged[selected_col]

    # Различные алгоритмы используют разные
    # кодировки. Не предполагаем автоматически,
    # что 1 всегда означает аномалию.
    unique_values = sorted(
        values.dropna().astype(str).unique().tolist()
    )

    anomaly_value = st.selectbox(
        "Какое значение обозначает аномалию?",
        unique_values,
        index=(
            unique_values.index("-1")
            if "-1" in unique_values else
            unique_values.index("1")
            if "1" in unique_values else 0
        )
    )

    anomaly_mask = (
        values.astype(str) == anomaly_value
    )

    marked = merged.loc[anomaly_mask]

    c1, c2 = st.columns(2)
    c1.metric("Аномальных наблюдений", len(marked))
    c2.metric(
        "Доля аномалий",
        f"{100 * len(marked) / len(merged):.2f}%"
    )

    fig = temperature_chart(
        merged,
        "Температурный ряд и аномалии"
    )

    fig.add_trace(
        go.Scatter(
            x=marked["date"],
            y=marked[TARGET],
            mode="markers",
            name="Аномалии",
            marker=dict(
                color="red",
                size=9,
                symbol="x"
            )
        )
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

    st.dataframe(
        marked[["date", TARGET, selected_col]],
        use_container_width=True
    )

# ------------------------------------------------------------
# 3. КЛАСТЕРИЗАЦИЯ
# ------------------------------------------------------------

elif section == "Кластеризация":

    st.header("Кластеризация временных окон")

    windows = load_optional(
        DATA_DIR / "task4_clustered_windows.csv"
    )

    if windows is None:
        st.warning(
            "Не найден файл task4_clustered_windows.csv"
        )
        st.stop()

    windows = normalize_dates(
        windows, ["start_date", "end_date"]
    )

    methods = {
        "Кластеризация по признакам": "cluster_feature",
        "Shape-based DTW": "cluster_shape",
        "PCA + KMeans": "cluster_pca"
    }

    available = {
        name: column
        for name, column in methods.items()
        if column in windows.columns
    }

    if not available:
        st.error("Столбцы кластеризации не найдены.")
        st.stop()

    selected_method = st.selectbox(
        "Метод кластеризации",
        list(available.keys())
    )

    cluster_col = available[selected_method]

    selected_windows = windows[
        (windows["end_date"] >= pd.Timestamp(start_date))
        & (
            windows["start_date"]
            <= pd.Timestamp(end_date)
        )
    ].copy()

    if selected_windows.empty:
        st.info("Для выбранного периода нет окон.")
        st.stop()

    st.subheader("Распределение временных окон")

    counts = (
        selected_windows[cluster_col]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="cluster",
        y="count",
        labels={
            "cluster": "Кластер",
            "count": "Количество окон"
        }
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

    # Для отображения каждого дня используем
    # интервал соответствующего 30-дневного окна.
    pieces = []

    for _, window in selected_windows.iterrows():

        part = filtered[
            filtered["date"].between(
                window["start_date"],
                window["end_date"]
            )
        ].copy()

        part["cluster"] = str(
            window[cluster_col]
        )

        pieces.append(part)

    if pieces:

        cluster_daily = pd.concat(
            pieces, ignore_index=True
        )

        st.subheader(
            "Температура с окраской по кластерам"
        )

        fig = px.scatter(
            cluster_daily,
            x="date",
            y=TARGET,
            color="cluster",
            labels={
                "date": "Дата",
                TARGET: "Температура, °C",
                "cluster": "Кластер"
            }
        )

        fig.update_traces(marker_size=6)

        st.plotly_chart(
            fig, use_container_width=True
        )

        cluster_stats = (
            cluster_daily
            .groupby("cluster")[TARGET]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )

        st.subheader("Характеристики кластеров")

        st.dataframe(
            cluster_stats.round(3),
            use_container_width=True
        )

    st.subheader("Таблица временных окон")

    st.dataframe(
        selected_windows[
            ["start_date", "end_date", cluster_col]
        ],
        use_container_width=True
    )

# ------------------------------------------------------------
# 4. СЕГМЕНТАЦИЯ
# ------------------------------------------------------------

elif section == "Сегментация":

    st.header("Сегментация временного ряда")

    segments = load_optional(
        REPORTS_DIR / "task5_segments_stationarity.csv"
    )

    if segments is None:
        st.warning(
            "Не найден файл task5_segments_stationarity.csv"
        )
        st.stop()

    segments = normalize_dates(
        segments, ["start_date", "end_date"]
    )

    required = [
        "segment", "start_date", "end_date"
    ]

    missing = [
        c for c in required
        if c not in segments.columns
    ]

    if missing:
        st.error(
            "Отсутствуют столбцы: "
            + ", ".join(missing)
        )
        st.stop()

    selected_segments = segments[
        (
            segments["end_date"]
            >= pd.Timestamp(start_date)
        )
        & (
            segments["start_date"]
            <= pd.Timestamp(end_date)
        )
    ].copy()

    st.metric(
        "Сегментов в выбранном периоде",
        len(selected_segments)
    )

    fig = temperature_chart(
        filtered,
        "Температурный ряд и границы режимов"
    )

    for _, segment in selected_segments.iterrows():

        boundary = segment["start_date"]

        if (
            boundary >= pd.Timestamp(start_date)
            and boundary <= pd.Timestamp(end_date)
        ):
            fig.add_vline(
                x=boundary,
                line_dash="dash",
                line_color="red",
                opacity=0.6
            )

    st.plotly_chart(
        fig, use_container_width=True
    )

    st.subheader("Характеристики сегментов")

    segment_rows = []

    for _, segment in selected_segments.iterrows():

        part = filtered[
            filtered["date"].between(
                segment["start_date"],
                segment["end_date"]
            )
        ]

        if part.empty:
            continue

        segment_rows.append({
            "segment": segment["segment"],
            "start_date": segment["start_date"],
            "end_date": segment["end_date"],
            "n": len(part),
            "mean": part[TARGET].mean(),
            "std": part[TARGET].std(),
            "min": part[TARGET].min(),
            "max": part[TARGET].max()
        })

    if segment_rows:

        stats = pd.DataFrame(segment_rows)

        st.dataframe(
            stats.round(3),
            use_container_width=True
        )

        fig = px.bar(
            stats,
            x="segment",
            y="mean",
            labels={
                "segment": "Номер сегмента",
                "mean": "Средняя температура, °C"
            }
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    st.caption(
        "Границы сегментов получены ранее методом PELT "
        "на полном временном ряду и предназначены "
        "для ретроспективного анализа."
    )

# ------------------------------------------------------------
# ЗАГРУЗКА ОТФИЛЬТРОВАННЫХ ДАННЫХ
# ------------------------------------------------------------

st.divider()

st.download_button(
    label="Сохранить выбранный период в CSV",
    data=filtered.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="temperature_filtered.csv",
    mime="text/csv"
)

st.caption(
    "Практическая работа №2 · "
    "Интерактивный структурный анализ данных"
)
