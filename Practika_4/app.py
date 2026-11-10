
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

PROJECT = Path(__file__).resolve().parent

DATA_DIR = PROJECT / "data"
MODEL_DIR = PROJECT / "models" / "task4"

DATE_COL = "date"
TARGET = "temperature_2m_mean"

MODEL_PATHS = {
    "Lasso": MODEL_DIR / "Lasso.joblib",
    "ExtraTrees": MODEL_DIR / "ExtraTrees.joblib"
}

st.set_page_config(
    page_title="Прогноз температуры Москвы",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. СТИЛИ
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_builtin_data():

    path = DATA_DIR / "task3_test.csv"

    return pd.read_csv(
        path,
        parse_dates=[DATE_COL]
    )


# ============================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_feature_names(model):

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(model, Pipeline):

        if hasattr(
            model.steps[0][1],
            "feature_names_in_"
        ):
            return list(
                model.steps[0][1].feature_names_in_
            )

    raise ValueError(
        "Не удалось определить признаки модели."
    )


def get_final_model(model):

    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


def get_importances(model, features):

    final_model = get_final_model(model)

    if hasattr(final_model, "coef_"):

        values = np.abs(
            np.asarray(final_model.coef_).ravel()
        )

        label = "Абсолютный коэффициент"

    elif hasattr(
        final_model,
        "feature_importances_"
    ):

        values = np.asarray(
            final_model.feature_importances_
        ).ravel()

        label = "Важность признака"

    else:
        return None, None

    if len(values) != len(features):
        return None, None

    importance = pd.DataFrame({
        "Признак": features,
        "Важность": values
    })

    importance = importance.sort_values(
        "Важность",
        ascending=False
    )

    return importance, label


def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "R2": r2_score(
            actual,
            predicted
        )
    }


def prepare_data(df, features):

    df = df.copy()

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Отсутствуют признаки: "
            + ", ".join(missing)
        )

    if DATE_COL in df.columns:

        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL],
            errors="coerce"
        )

        if df[DATE_COL].isna().any():
            raise ValueError(
                "Не удалось распознать некоторые даты."
            )

        df = df.sort_values(
            DATE_COL
        ).reset_index(drop=True)

        if df[DATE_COL].duplicated().any():
            raise ValueError(
                "Обнаружены повторяющиеся даты."
            )

    for col in features:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[features].isna().any().any():
        raise ValueError(
            "В признаках обнаружены пропущенные "
            "или нечисловые значения."
        )

    if not np.isfinite(
        df[features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Признаки содержат бесконечные значения."
        )

    if TARGET in df.columns:

        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )

        if (
            df[TARGET].isna().any()
            or not np.isfinite(
                df[TARGET].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                "Целевая переменная содержит "
                "некорректные значения."
            )

    return df


# ============================================================
# 5. ЗАГОЛОВОК
# ============================================================

st.title("🌡️ Прогноз температуры воздуха в Москве")

st.caption(
    "Интерактивное приложение для анализа "
    "результатов машинного обучения"
)

st.markdown(
    "Приложение использует обученные модели "
    "**Lasso** и **ExtraTrees** для прогнозирования "
    "среднесуточной температуры."
)


# ============================================================
# 6. БОКОВАЯ ПАНЕЛЬ
# ============================================================

st.sidebar.header("⚙️ Настройки")

model_name = st.sidebar.selectbox(
    "Модель машинного обучения",
    list(MODEL_PATHS.keys())
)

source = st.sidebar.radio(
    "Источник данных",
    [
        "Встроенная тестовая выборка",
        "Загрузить CSV"
    ]
)

model_path = MODEL_PATHS[model_name]

if not model_path.exists():

    st.error(
        f"Не найден файл модели: {model_path}"
    )

    st.stop()

try:

    model = load_model(
        str(model_path)
    )

    features = get_feature_names(
        model
    )

except Exception as error:

    st.error(
        f"Ошибка загрузки модели: {error}"
    )

    st.stop()


# ============================================================
# 7. ЗАГРУЗКА ДАННЫХ
# ============================================================

if source == "Встроенная тестовая выборка":

    try:

        df = load_builtin_data()

    except Exception as error:

        st.error(
            f"Ошибка загрузки данных: {error}"
        )

        st.stop()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Выберите CSV-файл",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Загрузите CSV-файл для продолжения."
        )

        st.stop()

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Не удалось прочитать CSV: {error}"
        )

        st.stop()


try:

    df = prepare_data(
        df,
        features
    )

except ValueError as error:

    st.error(str(error))
    st.stop()


# ============================================================
# 8. ФИЛЬТРАЦИЯ ПО ДАТАМ
# ============================================================

if DATE_COL in df.columns:

    min_date = df[DATE_COL].min().date()
    max_date = df[DATE_COL].max().date()

    selected_dates = st.sidebar.date_input(
        "Период анализа",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        df = df[
            (
                df[DATE_COL].dt.date
                >= start_date
            )
            &
            (
                df[DATE_COL].dt.date
                <= end_date
            )
        ].copy()

if df.empty:

    st.warning(
        "За выбранный период нет наблюдений."
    )

    st.stop()


# ============================================================
# 9. ПРОГНОЗИРОВАНИЕ
# ============================================================

X = df[features].astype(float)

try:

    predictions = np.asarray(
        model.predict(X),
        dtype=float
    ).ravel()

except Exception as error:

    st.error(
        f"Ошибка прогнозирования: {error}"
    )

    st.stop()

if (
    len(predictions) != len(df)
    or not np.isfinite(predictions).all()
):

    st.error(
        "Модель вернула некорректные прогнозы."
    )

    st.stop()

result = df.copy()

result["prediction"] = predictions

has_actual = TARGET in result.columns

if has_actual:

    result["residual"] = (
        result[TARGET]
        - result["prediction"]
    )

    result["absolute_error"] = (
        result["residual"].abs()
    )


# ============================================================
# 10. ИНФОРМАЦИОННЫЕ ПОКАЗАТЕЛИ
# ============================================================

st.subheader("📊 Основные показатели")

if has_actual:

    metrics = calculate_metrics(
        result[TARGET],
        result["prediction"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "MAE",
        f"{metrics['MAE']:.3f} °C"
    )

    col2.metric(
        "RMSE",
        f"{metrics['RMSE']:.3f} °C"
    )

    col3.metric(
        "R²",
        f"{metrics['R2']:.3f}"
    )

    col4.metric(
        "Наблюдений",
        len(result)
    )

else:

    col1, col2 = st.columns(2)

    col1.metric(
        "Наблюдений",
        len(result)
    )

    col2.metric(
        "Средний прогноз",
        f"{result['prediction'].mean():.2f} °C"
    )

    st.info(
        "Фактическая температура отсутствует. "
        "Метрики качества недоступны."
    )


# ============================================================
# 11. ГРАФИК ФАКТИЧЕСКОЙ И ПРОГНОЗНОЙ ТЕМПЕРАТУРЫ
# ============================================================

st.subheader("📈 Динамика температуры")

if DATE_COL in result.columns:

    x_axis = result[DATE_COL]

else:

    x_axis = np.arange(
        len(result)
    )

fig = go.Figure()

if has_actual:

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=result[TARGET],
            mode="lines",
            name="Фактическая температура",
            line=dict(width=2)
        )
    )

fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=result["prediction"],
        mode="lines",
        name="Прогноз модели",
        line=dict(width=2)
    )
)

fig.update_layout(
    xaxis_title="Дата / наблюдение",
    yaxis_title="Температура, °C",
    hovermode="x unified",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 12. ГРАФИК ОСТАТКОВ
# ============================================================

if has_actual:

    st.subheader("📉 Анализ остатков")

    fig_res = go.Figure()

    fig_res.add_trace(
        go.Scatter(
            x=x_axis,
            y=result["residual"],
            mode="lines",
            name="Остатки"
        )
    )

    fig_res.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig_res.update_layout(
        xaxis_title="Дата / наблюдение",
        yaxis_title="Факт − прогноз, °C",
        height=350
    )

    st.plotly_chart(
        fig_res,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Средний остаток",
        f"{result['residual'].mean():.3f} °C"
    )

    col2.metric(
        "Стандартное отклонение",
        f"{result['residual'].std():.3f} °C"
    )

    col3.metric(
        "Максимальная ошибка",
        f"{result['absolute_error'].max():.3f} °C"
    )


# ============================================================
# 13. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================

st.subheader("🔍 Важность признаков")

importance, importance_label = (
    get_importances(
        model,
        features
    )
)

if importance is not None:

    top_n = st.slider(
        "Количество признаков",
        min_value=5,
        max_value=min(
            30,
            len(importance)
        ),
        value=min(
            10,
            len(importance)
        )
    )

    top_features = importance.head(
        top_n
    ).sort_values("Важность")

    fig_importance = px.bar(
        top_features,
        x="Важность",
        y="Признак",
        orientation="h",
        title=f"{model_name}: важность признаков"
    )

    fig_importance.update_layout(
        height=450,
        xaxis_title=importance_label
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    if model_name == "Lasso":

        st.caption(
            "Для Lasso показаны абсолютные значения "
            "коэффициентов стандартизированных признаков."
        )

    else:

        st.caption(
            "Для ExtraTrees показана встроенная "
            "важность признаков."
        )

else:

    st.info(
        "Для выбранной модели важность "
        "признаков недоступна."
    )


# ============================================================
# 14. ТАБЛИЦА ПРОГНОЗОВ
# ============================================================

st.subheader("📋 Результаты прогнозирования")

columns = []

if DATE_COL in result.columns:
    columns.append(DATE_COL)

if has_actual:
    columns.append(TARGET)

columns.append("prediction")

if has_actual:
    columns.extend([
        "residual",
        "absolute_error"
    ])

st.dataframe(
    result[columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 15. СКАЧИВАНИЕ CSV
# ============================================================

csv_data = (
    result[columns]
    .to_csv(
        index=False
    )
    .encode("utf-8-sig")
)

st.download_button(
    label="⬇️ Скачать прогнозы CSV",
    data=csv_data,
    file_name=(
        f"temperature_predictions_"
        f"{model_name}.csv"
    ),
    mime="text/csv"
)


# ============================================================
# 16. ИНФОРМАЦИЯ О ПРОЕКТЕ
# ============================================================

with st.expander("ℹ️ Информация о приложении"):

    st.markdown(
        """
        **Проект:** прогнозирование среднесуточной
        температуры воздуха в Москве.

        **Модели:** Lasso, ExtraTrees.

        **Данные:** подготовленные признаки
        временного ряда.

        **Метрики:** MAE, RMSE, R².

        **Инструменты:** Python, scikit-learn,
        pandas, Plotly, Streamlit.

        Приложение выполняет прогнозирование
        по заранее рассчитанным признакам.
        """
    )
