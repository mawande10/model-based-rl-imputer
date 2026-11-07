
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# TOP 5 AFRICAN ECONOMIES
# GDP VALUES (USD BILLIONS)
# =====================================

TOP5_AFRICA = [
    "Nigeria",
    "South Africa",
    "Egypt",
    "Algeria",
    "Morocco"
]

TOP5_GDP = {

    "Nigeria": {
        2015: 493,
        2016: 404,
        2017: 375,
        2018: 421,
        2019: 448,
        2020: 432,
        2021: 441,
        2022: 477,
        2023: 363,
        2024: 390
    },

    "South Africa": {
        2015: 346,
        2016: 323,
        2017: 381,
        2018: 405,
        2019: 389,
        2020: 337,
        2021: 419,
        2022: 406,
        2023: 377,
        2024: 399
    },

    "Egypt": {
        2015: 332,
        2016: 332,
        2017: 235,
        2018: 250,
        2019: 303,
        2020: 365,
        2021: 424,
        2022: 476,
        2023: 398,
        2024: 430
    },

    "Algeria": {
        2015: 165,
        2016: 160,
        2017: 170,
        2018: 175,
        2019: 171,
        2020: 145,
        2021: 163,
        2022: 191,
        2023: 245,
        2024: 260
    },

    "Morocco": {
        2015: 110,
        2016: 111,
        2017: 118,
        2018: 127,
        2019: 129,
        2020: 121,
        2021: 142,
        2022: 134,
        2023: 145,
        2024: 155
    }
}

st.set_page_config(
    page_title="Model-Based RL + Online Optimization",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Model-Based RL + Online Optimization")
st.write(
    "Upload an Excel file, detect annual columns automatically, "
    "impute missing values, validate the model, and download the result."
)

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("Settings")
start_year = st.sidebar.number_input(
    "Start year", min_value=1900, max_value=2100, value=2015, step=1
)
end_year = st.sidebar.number_input(
    "End year", min_value=1900, max_value=2100, value=2024, step=1
   
)
if start_year >= end_year:
    st.sidebar.error(
        "End year must be greater than start year."
    )
    st.stop()

if (end_year - start_year + 1) != 10:
    st.sidebar.error(
        "This model requires exactly 10 years (2015-2024)."
    )
    st.stop()
    
n_estimators = st.sidebar.slider(
    "Random Forest trees", min_value=100, max_value=1000, value=500, step=50
)
alpha = st.sidebar.slider(
    "Online optimization α", min_value=0.01, max_value=0.50, value=0.08, step=0.01
)
episodes = st.sidebar.slider(
    "Optimization iterations", min_value=10, max_value=500, value=100, step=10
)

# -----------------------------
# HELPERS
# -----------------------------

def detect_country_column(columns):

    candidates = [
        "geoUnit",
        "country",
        "Country",
        "country_name",
        "Country Name",
        "Entity"
    ]

    for col in candidates:
        if col in columns:
            return col

    return None


def load_excel(uploaded_file):

    df = pd.read_excel(uploaded_file)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    country_col = detect_country_column(
        df.columns
    )

    if country_col is None:

        raise ValueError(
            "Country column not detected."
        )

    df[country_col] = (
        df[country_col]
        .astype(str)
        .str.strip()
    )

    year_cols = sorted(
        [
            c for c in df.columns
            if (
                c.isdigit()
                and start_year <= int(c) <= end_year
            )
        ],
        key=int
    )

    if not year_cols:

        raise ValueError(
            f"No year columns found "
            f"between {start_year} and {end_year}"
        )

    for col in year_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return (
        df,
        country_col,
        year_cols
    )


def estimate_country_gdp(history):

    history = np.asarray(
        history,
        dtype=float
    )

    mean_value = np.nanmean(history)

    if not np.isfinite(mean_value):
        return 150.0

    if mean_value > 200:
        return 450.0

    elif mean_value > 100:
        return 250.0

    elif mean_value > 50:
        return 150.0

    return 75.0


def train_world_model(
    df,
    country_col,
    year_cols
):

    X = []
    y = []

    training_df = df[
        df[country_col].isin(
            TOP5_AFRICA
        )
    ]

    for _, row in training_df.iterrows():

        country = row[country_col]

        values = row[
            year_cols
        ].to_numpy(
            dtype=np.float64
        )

        for j in range(
            3,
            len(values)
        ):

            state = values[j-3:j]
            target = values[j]

            years_used = [
                int(year_cols[j-3]),
                int(year_cols[j-2]),
                int(year_cols[j-1])
            ]

            gdp_features = []

            for yr in years_used:

                gdp_value = (
                    TOP5_GDP
                    .get(country, {})
                    .get(yr, np.nan)
                )

                gdp_features.append(
                    gdp_value
                )

            growth = (
                (
                    gdp_features[-1]
                    -
                    gdp_features[-2]
                )
                /
                max(
                    gdp_features[-2],
                    1
                )
            )

            features = np.concatenate([
                state,
                gdp_features,
                [growth]
            ])

            if (
                np.all(
                    np.isfinite(features)
                )
                and
                np.isfinite(target)
            ):

                X.append(features)
                y.append(target)

    if len(X) < 10:
        return None, 0

    X = np.asarray(
        X,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return (
        model,
        len(X)
    )


def nearest_neighbor_prediction(
    values,
    position,
    global_mean
):

    left = None
    right = None

    for k in range(
        position - 1,
        -1,
        -1
    ):

        if np.isfinite(values[k]):

            left = float(values[k])
            break

    for k in range(
        position + 1,
        len(values)
    ):

        if np.isfinite(values[k]):

            right = float(values[k])
            break

    if left is not None and right is not None:
        return (left + right) / 2

    if left is not None:
        return left

    if right is not None:
        return right

    row_mean = np.nanmean(values)

    if np.isfinite(row_mean):
        return float(row_mean)

    return float(global_mean)


def safe_prediction(
    model,
    state
):

    state = np.asarray(
        state,
        dtype=np.float64
    )

    if len(state) != 3:
        return np.nan

    estimated_gdp = (
        estimate_country_gdp(
            state
        )
    )

    growth = 0.03

    features = np.concatenate([
        state,
        [
            estimated_gdp,
            estimated_gdp,
            estimated_gdp,
            growth
        ]
    ])

    pred = model.predict(
        features.reshape(1, -1)
    )

    pred = np.asarray(
        pred
    ).reshape(-1)

    if pred.size == 0:
        return np.nan

    value = float(pred[0])

    value = max(
        value,
        0
    )

    if np.isfinite(value):
        return value

    return np.nan


def impute_dataframe(df, year_cols, model):
    result = df.copy()

    # IMPORTANT:
    # Preserve exactly which cells were missing in the user's upload.
    original_missing = result[year_cols].isna().copy()

    numeric_matrix = result[year_cols].to_numpy(
        dtype=np.float64, copy=True
    )

    global_mean = np.nanmean(numeric_matrix)

    if not np.isfinite(global_mean):
        global_mean = 0.0

    for row_number in range(len(result)):
        # Explicit float array prevents:
        # "setting an array element with a sequence"
        values = np.asarray(
            numeric_matrix[row_number],
            dtype=np.float64
        ).copy()

        missing_mask = np.asarray(
            original_missing.iloc[row_number].to_numpy(),
            dtype=bool
        )

        if not missing_mask.any():
            continue

        for _ in range(episodes):
            previous = values.copy()

            for j in range(len(values)):
                # Never modify an originally observed value.
                if not missing_mask[j]:
                    continue

                prediction = np.nan

                # -----------------------------
                # WORLD MODEL
                # -----------------------------
                if model is not None and j >= 3:
                    state = values[j - 3:j]

                    if np.all(np.isfinite(state)):
                        prediction = safe_prediction(model, state)

                # -----------------------------
                # FALLBACK
                # -----------------------------
                if not np.isfinite(prediction):
                    prediction = nearest_neighbor_prediction(
                        values, j, global_mean
                    )

                if not np.isfinite(prediction):
                    prediction = float(global_mean)

                # -----------------------------
                # MODEL-BASED RL + ONLINE UPDATE
                # -----------------------------
                if not np.isfinite(values[j]):
                    values[j] = float(prediction)
                else:
                    updated = (
                        values[j]
                        + alpha * (float(prediction) - values[j])
                    )

                    if np.isfinite(updated):
                        values[j] = float(updated)

            finite_values = values[np.isfinite(values)]

            if len(finite_values) == len(values):
                break

            if np.all(np.isfinite(previous)) and np.all(np.isfinite(values)):
                diff = np.max(np.abs(values - previous))
                if diff < 1e-6:
                    break

        numeric_matrix[row_number] = values

    result.loc[:, year_cols] = np.round(numeric_matrix, 3)

    return result


def validation(
    df,
    country_col,
    year_cols,
    model
):

    if model is None:
        return None

    actual = []
    predicted = []

    validation_df = df[
        df[country_col].isin(
            TOP5_AFRICA
        )
    ]

    for _, row in validation_df.iterrows():

        country = row[country_col]

        values = row[
            year_cols
        ].to_numpy(
            dtype=np.float64
        )

        for j in range(
            3,
            len(values)
        ):

            state = values[j-3:j]
            target = values[j]

            if (
                not np.all(
                    np.isfinite(state)
                )
                or
                not np.isfinite(target)
            ):
                continue

            years_used = [
                int(year_cols[j-3]),
                int(year_cols[j-2]),
                int(year_cols[j-1])
            ]

            gdp_features = []

            for yr in years_used:

                gdp_value = (
                    TOP5_GDP
                    .get(country, {})
                    .get(yr, np.nan)
                )

                gdp_features.append(
                    gdp_value
                )

            if not np.all(
                np.isfinite(gdp_features)
            ):
                continue

            growth = (
                (
                    gdp_features[-1]
                    -
                    gdp_features[-2]
                )
                /
                max(
                    gdp_features[-2],
                    1
                )
            )

            features = np.concatenate([
                state,
                gdp_features,
                [growth]
            ])

            pred = model.predict(
                features.reshape(1, -1)
            )[0]

            if np.isfinite(pred):

                actual.append(
                    float(target)
                )

                predicted.append(
                    float(pred)
                )

    if len(actual) < 2:
        return None

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = (
        r2_score(
            actual,
            predicted
        )
        if len(set(actual)) > 1
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R²": float(r2),
        "Validation samples": len(actual)
    }


# -----------------------------
# APPLICATION
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your Excel file",
    type=["xlsx", "xls"]
)

if uploaded_file is None:

    st.info(
        "Upload an Excel file to begin."
    )

    st.markdown(
        """
        **Expected format**

        geoUnit | 2015 | 2016 | ... | 2024

        Only missing values will be imputed.
        Existing values are preserved.
        """
    )

else:

    try:

        df, country_col, year_cols = load_excel(
            uploaded_file
        )

        before_missing = int(
            df[year_cols]
            .isna()
            .sum()
            .sum()
        )

        st.success(
            f"File loaded: {uploaded_file.name}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(df)
        )

        c2.metric(
            "Detected years",
            len(year_cols)
        )

        c3.metric(
            "Missing values",
            before_missing
        )

        st.write(
            "**Country Column:**",
            country_col
        )

        st.write(
            "**Detected Years:**",
            ", ".join(year_cols)
        )

        # --------------------------------
        # Missing values by year
        # --------------------------------

        st.subheader(
            "Missing Values by Year"
        )

        missing_by_year = (
            df[year_cols]
            .isna()
            .sum()
        )

        st.bar_chart(
            missing_by_year
        )

        with st.expander(
            "View Uploaded Dataset"
        ):

            st.dataframe(
                df,
                use_container_width=True
            )

        # --------------------------------
        # RUN MODEL
        # --------------------------------

        if st.button(
            "🚀 Run GDP-Aware Model",
            type="primary"
        ):

            with st.spinner(
                "Training GDP-aware world model..."
            ):

                world_model, training_samples = (
                    train_world_model(
                        df,
                        country_col,
                        year_cols
                    )
                )

            if world_model is None:

                st.error(
                    "Not enough training samples "
                    "to build the GDP world model."
                )

                st.stop()

            st.success(
                "GDP World Model Trained"
            )

            st.write(
                "Training Countries:"
            )

            st.write(
                ", ".join(
                    TOP5_AFRICA
                )
            )

            st.info(
                f"Training samples: "
                f"{training_samples:,}"
            )

            # --------------------------------
            # FEATURE IMPORTANCE
            # --------------------------------

            feature_names = [
                "Education_t-3",
                "Education_t-2",
                "Education_t-1",
                "GDP_t-3",
                "GDP_t-2",
                "GDP_t-1",
                "GDP_Growth"
            ]

            importance_df = pd.DataFrame(
                {
                    "Feature":
                    feature_names,
                    "Importance":
                    world_model.feature_importances_
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            st.subheader(
                "Feature Importance"
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            # --------------------------------
            # IMPUTE DATA
            # --------------------------------

            with st.spinner(
                "Imputing missing values..."
            ):

                df_imputed = (
                    impute_dataframe(
                        df,
                        year_cols,
                        world_model
                    )
                )

            after_missing = int(
                df_imputed[year_cols]
                .isna()
                .sum()
                .sum()
            )

            filled = (
                before_missing
                - after_missing
            )

            st.subheader(
                "Results"
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Missing Before",
                before_missing
            )

            r2.metric(
                "Missing After",
                after_missing
            )

            r3.metric(
                "Values Filled",
                filled
            )

            if after_missing == 0:

                st.success(
                    "All missing values "
                    "were successfully filled."
                )

            else:

                st.warning(
                    f"{after_missing} values "
                    "could not be imputed."
                )

            # --------------------------------
            # IMPUTED DATASET
            # --------------------------------

            st.subheader(
                "Imputed Dataset"
            )

            st.dataframe(
                df_imputed,
                use_container_width=True,
                height=600
            )

            # --------------------------------
            # VALIDATION
            # --------------------------------

            metrics = validation(
                df,
                country_col,
                year_cols,
                world_model
            )

            if metrics is not None:

                st.subheader(
                    "Validation Metrics"
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "MAE",
                    f"{metrics['MAE']:.4f}"
                )

                m2.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.4f}"
                )

                if np.isfinite(
                    metrics["R²"]
                ):

                    m3.metric(
                        "R²",
                        f"{metrics['R²']:.4f}"
                    )

                else:

                    m3.metric(
                        "R²",
                        "N/A"
                    )

                m4.metric(
                    "Samples",
                    metrics[
                        "Validation samples"
                    ]
                )

            # --------------------------------
            # EXPORT
            # --------------------------------

            output = io.BytesIO()

            with pd.ExcelWriter(
                output,
                engine="openpyxl"
            ) as writer:

                df.to_excel(
                    writer,
                    sheet_name="Original_Data",
                    index=False
                )

                df_imputed.to_excel(
                    writer,
                    sheet_name="Imputed_Data",
                    index=False
                )

                summary_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Missing Before",
                            "Missing After",
                            "Values Filled",
                            "Training Samples",
                            "Start Year",
                            "End Year",
                            "RandomForest Trees",
                            "Alpha",
                            "Episodes"
                        ],

                        "Value": [
                            before_missing,
                            after_missing,
                            filled,
                            training_samples,
                            start_year,
                            end_year,
                            n_estimators,
                            alpha,
                            episodes
                        ]
                    }
                )

                summary_df.to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False
                )

            output.seek(0)

            st.download_button(
                label="⬇ Download Results",
                data=output.getvalue(),
                file_name=(
                    "GDP_Aware_"
                    "Education_Imputed.xlsx"
                ),
                mime=(
                    "application/"
                    "vnd.openxmlformats-"
                    "officedocument."
                    "spreadsheetml.sheet"
                )
            )

            st.success(
                "Processing complete. "
                "Download the Excel file above."
            )

    except Exception as e:

        st.error(
            "The uploaded file could "
            "not be processed."
        )

        st.exception(e)
