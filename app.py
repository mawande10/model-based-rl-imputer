
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Model-Based RL + Online Optimization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model-Based RL + Online Optimization")
st.markdown(
    "Upload an Excel dataset, automatically detect country/year columns, "
    "impute missing annual values, and download the completed Excel file."
)

st.sidebar.header("Settings")
start_year = st.sidebar.number_input("Start year", min_value=1900, max_value=2100, value=2015, step=1)
end_year = st.sidebar.number_input("End year", min_value=1900, max_value=2100, value=2025, step=1)
n_estimators = st.sidebar.slider("Random Forest trees", 100, 1000, 500, 50)
alpha = st.sidebar.slider("Online optimization α", 0.01, 0.50, 0.08, 0.01)
episodes = st.sidebar.slider("Optimization iterations", 10, 500, 100, 10)

uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx or .xls)",
    type=["xlsx", "xls"]
)

def detect_country_column(columns):
    candidates = ["geoUnit", "country", "Country", "country_name", "Country Name", "Entity"]
    for c in candidates:
        if c in columns:
            return c
    return None

def prepare_dataframe(file):
    df = pd.read_excel(file)
    df.columns = df.columns.astype(str).str.strip()

    country_col = detect_country_column(df.columns)
    if country_col is None:
        st.error(
            "Could not identify the country column. "
            "Expected one of: geoUnit, country, Country, country_name, Country Name, Entity."
        )
        st.stop()

    year_cols = sorted(
        [
            c for c in df.columns
            if c.isdigit() and start_year <= int(c) <= end_year
        ],
        key=int
    )

    if not year_cols:
        st.error(
            f"No year columns were detected between {start_year} and {end_year}. "
            "Make sure your Excel file contains columns such as 2015, 2016, ..., 2025."
        )
        st.stop()

    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, country_col, year_cols

def build_world_model(df, year_cols, n_estimators):
    X, y = [], []

    # Learn temporal transitions: [t-3, t-2, t-1] -> t
    for _, row in df.iterrows():
        values = row[year_cols].to_numpy(dtype=float)

        for i in range(3, len(values)):
            state = values[i-3:i]
            target = values[i]

            if not np.isnan(state).any() and not np.isnan(target):
                X.append(state)
                y.append(target)

    if len(X) < 10:
        return None, 0

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    )
    model.fit(X, y)
    return model, len(X)

def fallback_prediction(values, j, global_mean):
    left = None
    right = None

    for k in range(j - 1, -1, -1):
        if not np.isnan(values[k]):
            left = values[k]
            break

    for k in range(j + 1, len(values)):
        if not np.isnan(values[k]):
            right = values[k]
            break

    if left is not None and right is not None:
        return (left + right) / 2.0
    if left is not None:
        return left
    if right is not None:
        return right

    row_mean = np.nanmean(values)
    if not np.isnan(row_mean):
        return row_mean

    return global_mean

def impute_dataframe(df, year_cols, model, alpha, episodes):
    result = df.copy()
    original_missing = result[year_cols].isna()

    global_mean = np.nanmean(result[year_cols].to_numpy(dtype=float))
    if np.isnan(global_mean):
        global_mean = 0.0

    for idx in result.index:
        values = result.loc[idx, year_cols].to_numpy(dtype=float)
        missing_mask = original_missing.loc[idx].to_numpy(dtype=bool)

        if not missing_mask.any():
            continue

        for _ in range(episodes):
            previous = values.copy()

            for j in range(len(values)):
                if not missing_mask[j]:
                    continue

                prediction = np.nan

                # Model-based world-model prediction
                if model is not None and j >= 3:
                    state = values[j-3:j]
                    if not np.isnan(state).any():
                        prediction = float(model.predict(state.reshape(1, -1))[0])

                # Fallback for beginning gaps or unavailable state
                if np.isnan(prediction):
                    prediction = fallback_prediction(values, j, global_mean)

                if np.isnan(values[j]):
                    values[j] = prediction
                else:
                    # Online optimization update
                    values[j] = values[j] + alpha * (prediction - values[j])

            if np.all(~np.isnan(values)):
                break

            diff = np.nanmax(np.abs(values - previous))
            if not np.isfinite(diff) or diff < 1e-6:
                break

        result.loc[idx, year_cols] = np.round(values, 3)

    return result

def validation(df, year_cols, model, seed=42):
    # Mask a subset of observed interior values and predict them.
    rng = np.random.default_rng(seed)
    actual, predicted = [], []

    for _, row in df.iterrows():
        values = row[year_cols].to_numpy(dtype=float)

        candidate = [
            i for i in range(3, len(values))
            if not np.isnan(values[i])
            and not np.isnan(values[i-3:i]).any()
        ]

        if not candidate:
            continue

        n_test = max(1, int(0.2 * len(candidate)))
        test_idx = rng.choice(candidate, size=n_test, replace=False)

        for j in test_idx:
            state = values[j-3:j]
            pred = float(model.predict(state.reshape(1, -1))[0])
            actual.append(values[j])
            predicted.append(pred)

    if len(actual) < 2:
        return None

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted) if len(set(actual)) > 1 else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Validation samples": len(actual)
    }

if uploaded_file is not None:
    try:
        df, country_col, year_cols = prepare_dataframe(uploaded_file)

        st.success(f"File loaded: {uploaded_file.name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Year columns", len(year_cols))
        c3.metric("Missing values", int(df[year_cols].isna().sum().sum()))

        st.subheader("Input dataset")
        st.dataframe(df, use_container_width=True, height=400)

        st.write("**Detected country column:**", country_col)
        st.write("**Detected years:**", ", ".join(year_cols))

        if st.button("🚀 Run Model-Based RL + Online Optimization", type="primary"):
            before_missing = int(df[year_cols].isna().sum().sum())

            with st.spinner("Training world model..."):
                world_model, samples = build_world_model(
                    df, year_cols, n_estimators
                )

            if world_model is None:
                st.error(
                    "There are not enough complete temporal sequences to train "
                    "the Random Forest world model. At least 10 valid training "
                    "samples are recommended."
                )
                st.stop()

            st.info(f"World model trained using {samples:,} temporal training samples.")

            with st.spinner("Imputing missing values and running online optimization..."):
                df_imputed = impute_dataframe(
                    df, year_cols, world_model, alpha, episodes
                )

            after_missing = int(df_imputed[year_cols].isna().sum().sum())
            filled = before_missing - after_missing

            st.subheader("Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Missing BEFORE", before_missing)
            r2.metric("Missing AFTER", after_missing)
            r3.metric("Values filled", filled)

            st.subheader("Before vs After")
            st.dataframe(
                pd.concat(
                    [
                        df[[country_col] + year_cols].assign(Status="BEFORE"),
                        df_imputed[[country_col] + year_cols].assign(Status="AFTER")
                    ],
                    ignore_index=True
                ),
                use_container_width=True,
                height=500
            )

            if after_missing > 0:
                st.warning(
                    "Some values remain missing because the uploaded dataset "
                    "does not contain enough information to estimate them."
                )

            metrics = validation(df, year_cols, world_model)
            if metrics:
                st.subheader("Validation metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{metrics['MAE']:.4f}")
                m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                m3.metric("R²", f"{metrics['R²']:.4f}" if np.isfinite(metrics["R²"]) else "N/A")
                m4.metric("Validation samples", metrics["Validation samples"])

            # Downloadable Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Original_Data", index=False)
                df_imputed.to_excel(writer, sheet_name="Imputed_Data", index=False)

                summary = pd.DataFrame({
                    "Metric": [
                        "Original missing values",
                        "Remaining missing values",
                        "Values filled",
                        "Training samples",
                        "Alpha",
                        "Episodes",
                        "Random Forest trees"
                    ],
                    "Value": [
                        before_missing,
                        after_missing,
                        filled,
                        samples,
                        alpha,
                        episodes,
                        n_estimators
                    ]
                })
                summary.to_excel(writer, sheet_name="Summary", index=False)

            output.seek(0)

            st.download_button(
                label="⬇️ Download Imputed Excel File",
                data=output.getvalue(),
                file_name="Model_Based_RL_Online_Optimization_Imputed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success(
                "Processing completed. The original data and imputed data are "
                "included in the downloadable Excel workbook."
            )
else:
    st.info("Upload an Excel file above to begin.")
    st.markdown("""
### Expected Excel structure

The application works with a wide-format table such as:

| geoUnit | 2015 | 2016 | 2017 | ... | 2025 |
|---|---:|---:|---:|---:|---:|
| Angola | 10.2 | NaN | 11.4 | ... | 15.1 |
| Botswana | 8.1 | 8.5 | NaN | ... | 12.7 |

**Important:** Existing observed values are never overwritten. Only values that were missing in the uploaded file are changed.

The downloaded workbook contains:
- `Original_Data`
- `Imputed_Data`
- `Summary`
""")
