from pathlib import Path
import zipfile, py_compile, textwrap, os

app_path = Path("/mnt/data/app.py")

app_code = r'''
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
    "End year", min_value=1900, max_value=2100, value=2025, step=1
)
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
        "geoUnit", "country", "Country",
        "country_name", "Country Name", "Entity"
    ]
    for col in candidates:
        if col in columns:
            return col
    return None


def load_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str).str.strip()

    country_col = detect_country_column(df.columns)
    if country_col is None:
        raise ValueError(
            "Country column not detected. Expected one of: "
            "geoUnit, country, Country, country_name, Country Name, Entity."
        )

    year_cols = sorted(
        [
            c for c in df.columns
            if c.isdigit() and start_year <= int(c) <= end_year
        ],
        key=int
    )

    if not year_cols:
        raise ValueError(
            f"No year columns found between {start_year} and {end_year}."
        )

    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df, country_col, year_cols


def train_world_model(df, year_cols):
    X, y = [], []

    for _, row in df.iterrows():
        values = row[year_cols].to_numpy(dtype=float)

        # Temporal state:
        # [t-3, t-2, t-1] -> t
        for j in range(3, len(values)):
            state = values[j - 3:j]
            target = values[j]

            if np.all(np.isfinite(state)) and np.isfinite(target):
                X.append(state.tolist())
                y.append(float(target))

    if len(X) < 10:
        return None, 0

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model, len(X)


def nearest_neighbor_prediction(values, position, global_mean):
    left = None
    right = None

    for k in range(position - 1, -1, -1):
        if np.isfinite(values[k]):
            left = float(values[k])
            break

    for k in range(position + 1, len(values)):
        if np.isfinite(values[k]):
            right = float(values[k])
            break

    if left is not None and right is not None:
        return (left + right) / 2.0

    if left is not None:
        return left

    if right is not None:
        return right

    row_mean = np.nanmean(values)
    if np.isfinite(row_mean):
        return float(row_mean)

    return float(global_mean)


def safe_prediction(model, state):
    """Always return a scalar float or NaN."""
    pred = model.predict(np.asarray(state, dtype=np.float64).reshape(1, -1))
    pred = np.asarray(pred).reshape(-1)

    if pred.size == 0:
        return np.nan

    value = float(pred[0])

    return value if np.isfinite(value) else np.nan


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


def validation(df, year_cols, model):
    if model is None:
        return None

    rng = np.random.default_rng(42)
    actual = []
    predicted = []

    for _, row in df.iterrows():
        values = row[year_cols].to_numpy(dtype=float)

        candidates = [
            j for j in range(3, len(values))
            if np.isfinite(values[j])
            and np.all(np.isfinite(values[j - 3:j]))
        ]

        if not candidates:
            continue

        test_count = max(1, int(0.20 * len(candidates)))
        test_positions = rng.choice(
            candidates,
            size=min(test_count, len(candidates)),
            replace=False
        )

        for j in test_positions:
            pred = safe_prediction(model, values[j - 3:j])

            if np.isfinite(pred):
                actual.append(float(values[j]))
                predicted.append(float(pred))

    if len(actual) < 2:
        return None

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted) if len(set(actual)) > 1 else np.nan

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
    st.info("Upload an Excel file to begin.")
    st.markdown(
        """
        **Expected format**

        `geoUnit | 2015 | 2016 | ... | 2025`

        Existing values are preserved. Only cells that were missing in
        the uploaded file are imputed.
        """
    )
else:
    try:
        df, country_col, year_cols = load_excel(uploaded_file)

        before_missing = int(
            df[year_cols].isna().sum().sum()
        )

        st.success(f"File loaded: {uploaded_file.name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Detected years", len(year_cols))
        c3.metric("Missing values", before_missing)

        st.write("**Country column:**", country_col)
        st.write("**Detected years:**", ", ".join(year_cols))

        with st.expander("View uploaded dataset"):
            st.dataframe(df, use_container_width=True)

        if st.button(
            "🚀 Run Model-Based RL + Online Optimization",
            type="primary"
        ):
            with st.spinner("Training temporal world model..."):
                world_model, training_samples = train_world_model(
                    df, year_cols
                )

            if world_model is None:
                st.error(
                    "Not enough complete temporal sequences to train "
                    "the world model. At least 10 training samples are required."
                )
                st.stop()

            st.info(
                f"World model trained using "
                f"{training_samples:,} temporal training samples."
            )

            with st.spinner(
                "Imputing missing values and running online optimization..."
            ):
                df_imputed = impute_dataframe(
                    df, year_cols, world_model
                )

            after_missing = int(
                df_imputed[year_cols].isna().sum().sum()
            )
            filled = before_missing - after_missing

            st.subheader("Results")

            r1, r2, r3 = st.columns(3)
            r1.metric("Missing BEFORE", before_missing)
            r2.metric("Missing AFTER", after_missing)
            r3.metric("Values filled", filled)

            if after_missing == 0:
                st.success("All missing year values were successfully filled.")
            else:
                st.warning(
                    f"{after_missing} values remain missing because the "
                    "uploaded data did not provide enough usable information."
                )

            st.subheader("Imputed Dataset")
            st.dataframe(df_imputed, use_container_width=True, height=500)

            metrics = validation(
                df, year_cols, world_model
            )

            if metrics:
                st.subheader("Validation Metrics")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{metrics['MAE']:.4f}")
                m2.metric("RMSE", f"{metrics['RMSE']:.4f}")

                if np.isfinite(metrics["R²"]):
                    m3.metric("R²", f"{metrics['R²']:.4f}")
                else:
                    m3.metric("R²", "N/A")

                m4.metric(
                    "Validation samples",
                    metrics["Validation samples"]
                )

            # -----------------------------
            # DOWNLOAD EXCEL
            # -----------------------------
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

                summary_data = {
                    "Metric": [
                        "Original missing values",
                        "Remaining missing values",
                        "Values filled",
                        "Training samples",
                        "Start year",
                        "End year",
                        "Random Forest trees",
                        "Online optimization alpha",
                        "Optimization iterations"
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

                if metrics:
                    summary_data["Metric"].extend([
                        "Validation MAE",
                        "Validation RMSE",
                        "Validation R²",
                        "Validation samples"
                    ])

                    summary_data["Value"].extend([
                        metrics["MAE"],
                        metrics["RMSE"],
                        metrics["R²"],
                        metrics["Validation samples"]
                    ])

                pd.DataFrame(summary_data).to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False
                )

            output.seek(0)

            st.download_button(
                label="⬇️ Download Imputed Excel File",
                data=output.getvalue(),
                file_name=(
                    "Model_Based_RL_Online_Optimization_Imputed.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

            st.success(
                "Processing complete. Your original and imputed datasets "
                "are available in the Excel download."
            )

    except Exception as e:
        st.error(
            "The uploaded file could not be processed. "
            "Please check that it is a valid Excel file and that it contains "
            "a country column and year columns."
        )

        # Show a safe diagnostic to the user without exposing file contents.
        st.exception(e)
'''

app_path.write_text(app_code, encoding="utf-8")

# Verify Python syntax.
py_compile.compile(str(app_path), doraise=True)

req_path = Path("/mnt/data/requirements.txt")
req_path.write_text(
    "streamlit\npandas\nnumpy\nscikit-learn\nopenpyxl\nxlrd\n",
    encoding="utf-8"
)

readme_path = Path("/mnt/data/README.md")
readme_path.write_text(
    """# Model-Based RL + Online Optimization Excel Imputer

Public Streamlit application for uploading an Excel file, detecting annual
columns, imputing missing values, validating the temporal world model, and
downloading the completed workbook.

Expected format:
`geoUnit | 2015 | 2016 | ... | 2025`

Existing observed values are preserved. Only originally missing cells are
updated.

Deploy with Streamlit Community Cloud using `app.py` as the main file.
""",
    encoding="utf-8"
)

zip_path = Path("/mnt/data/model_based_rl_online_optimization_PUBLIC_FIXED_v2.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(app_path, arcname="app.py")
    z.write(req_path, arcname="requirements.txt")
    z.write(readme_path, arcname="README.md")

print("Fixed application created and syntax-checked successfully.")
print(zip_path)
