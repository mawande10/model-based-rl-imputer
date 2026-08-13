# ============================================================
# app.py
# AFRICA MODEL-BASED RL + ONLINE OPTIMIZATION IMPUTER
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="African Data Imputation",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🌍 African Country Data Imputation"
)

st.subheader(
    "Model-Based RL + Online Optimization"
)

st.markdown(
    """
    Upload an Excel dataset containing African country data.

    The application will:

    1. Detect missing African countries.
    2. Detect missing values for existing countries.
    3. Train a country-level World Model.
    4. Generate trajectories for completely missing countries.
    5. Impute missing years for existing countries.
    6. Preserve all observed values.
    7. Produce a complete African 2015–2025 dataset.
    8. Allow the completed Excel file to be downloaded.
    """
)


# ============================================================
# AFRICAN COUNTRY LIST
# ============================================================

ALL_AFRICA = {
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR",
    "CAF", "TCD", "COM", "COG", "COD", "CIV", "DJI", "EGY",
    "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN",
    "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI",
    "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
    "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN",
    "TZA", "TGO", "TUN", "UGA", "ZMB", "ZWE"
}

AFRICA_YEARS = [
    str(y)
    for y in range(2015, 2026)
]


# ============================================================
# FUNCTION 1
# DETECT COMPLETELY MISSING AFRICAN COUNTRIES
# ============================================================

def detect_missing_african_countries(
    df,
    country_col="geoUnit"
):
    """
    Compare countries in the uploaded dataset against
    the complete African ISO-3 country list.
    """

    if country_col not in df.columns:

        raise ValueError(
            f"Required country column "
            f"'{country_col}' was not found."
        )

    existing = set(
        df[country_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    missing = sorted(
        ALL_AFRICA - existing
    )

    return missing


# ============================================================
# FUNCTION 2
# VALIDATE YEAR COLUMNS
# ============================================================

def validate_year_columns(
    df,
    required_years=None
):

    if required_years is None:

        required_years = AFRICA_YEARS

    missing_year_columns = [
        year
        for year in required_years
        if year not in df.columns
    ]

    return missing_year_columns


# ============================================================
# FUNCTION 3
# TRAIN COUNTRY-LEVEL WORLD MODEL
# ============================================================

def train_country_world_model(
    df,
    country_col="geoUnit",
    year_cols=None,
    n_estimators=500
):
    """
    Train a temporal Random Forest World Model.

    State:

        [y(t-3), y(t-2), y(t-1)]

    Target:

        y(t)

    The model learns temporal relationships across
    all countries in the uploaded dataset.
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    data = df.copy()

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    for col in year_cols:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # --------------------------------------------------------
    # Create temporal training samples
    # --------------------------------------------------------

    for _, row in data.iterrows():

        values = row[
            year_cols
        ].to_numpy(
            dtype=float
        )

        for i in range(
            3,
            len(values)
        ):

            state = values[
                i-3:i
            ]

            target = values[i]

            # Complete state required
            if np.isnan(state).any():
                continue

            # Target required
            if np.isnan(target):
                continue

            X.append(state)
            y.append(target)

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # --------------------------------------------------------
    # Validate training samples
    # --------------------------------------------------------

    if len(X) < 5:

        raise ValueError(
            "Not enough complete temporal observations "
            "to train the world model."
        )

    # --------------------------------------------------------
    # Random Forest World Model
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model, len(X)


# ============================================================
# FUNCTION 4
# GENERATE TRAJECTORY FOR COMPLETELY MISSING COUNTRY
# ============================================================

def generate_missing_country_trajectory(
    world_model,
    reference_df,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Generate a complete 2015–2025 trajectory using
    Model-Based RL + Online Optimization.

    The initial state is estimated from available
    African-country observations.
    """

    data = reference_df[
        year_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Estimate initial 3-year state
    # --------------------------------------------------------

    initial_values = []

    for year in year_cols[:3]:

        column = data[
            year
        ].dropna()

        if len(column) > 0:

            initial_values.append(
                float(
                    column.median()
                )
            )

        else:

            initial_values.append(
                np.nan
            )

    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    numeric_data = data.to_numpy(
        dtype=float
    )

    if np.isnan(
        numeric_data
    ).all():

        global_mean = 0.0

    else:

        global_mean = float(
            np.nanmean(
                numeric_data
            )
        )

    initial_values = [
        global_mean
        if np.isnan(x)
        else x
        for x in initial_values
    ]

    # --------------------------------------------------------
    # Create trajectory
    # --------------------------------------------------------

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    # Initial three years
    values[:3] = initial_values

    # --------------------------------------------------------
    # MODEL-BASED RL
    # --------------------------------------------------------

    for j in range(
        3,
        len(year_cols)
    ):

        state = values[
            j-3:j
        ]

        if np.isnan(state).any():
            continue

        prediction = world_model.predict(
            state.reshape(
                1,
                -1
            )
        )

        prediction = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        values[j] = prediction

    # --------------------------------------------------------
    # ONLINE OPTIMIZATION
    # --------------------------------------------------------

    for episode in range(
        episodes
    ):

        previous = values.copy()

        for j in range(
            3,
            len(values)
        ):

            state = values[
                j-3:j
            ]

            if np.isnan(
                state
            ).any():

                continue

            prediction = world_model.predict(
                state.reshape(
                    1,
                    -1
                )
            )

            prediction = float(
                np.asarray(
                    prediction
                ).reshape(-1)[0]
            )

            values[j] = (
                values[j]
                + alpha
                * (
                    prediction
                    - values[j]
                )
            )

        difference = np.max(
            np.abs(
                values
                - previous
            )
        )

        if difference < 1e-6:

            break

    return np.round(
        values,
        3
    )


# ============================================================
# FUNCTION 5
# IMPUTE EXISTING COUNTRY WITH MISSING YEARS
# ============================================================

def impute_existing_country(
    row,
    world_model,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Impute missing years for an existing country.

    IMPORTANT:
    Observed values are never modified.
    """

    original_values = (
        pd.to_numeric(
            row[year_cols],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    values = original_values.copy()

    # Remember exactly which values were missing
    missing_mask = np.isnan(
        original_values
    )

    # --------------------------------------------------------
    # Initial filling
    # --------------------------------------------------------

    for j in range(
        len(values)
    ):

        if not missing_mask[j]:
            continue

        prediction = np.nan

        # ----------------------------------------------------
        # World Model
        # ----------------------------------------------------

        if j >= 3:

            state = values[
                j-3:j
            ]

            if not np.isnan(
                state
            ).any():

                prediction = float(
                    np.asarray(
                        world_model.predict(
                            state.reshape(
                                1,
                                -1
                            )
                        )
                    ).reshape(-1)[0]
                )

        # ----------------------------------------------------
        # Neighbour fallback
        # ----------------------------------------------------

        if np.isnan(
            prediction
        ):

            left = None
            right = None

            for k in range(
                j - 1,
                -1,
                -1
            ):

                if not np.isnan(
                    values[k]
                ):

                    left = values[k]
                    break

            for k in range(
                j + 1,
                len(values)
            ):

                if not np.isnan(
                    values[k]
                ):

                    right = values[k]
                    break

            if (
                left is not None
                and right is not None
            ):

                prediction = (
                    left + right
                ) / 2.0

            elif left is not None:

                prediction = left

            elif right is not None:

                prediction = right

        # ----------------------------------------------------
        # Row mean fallback
        # ----------------------------------------------------

        if np.isnan(
            prediction
        ):

            available = values[
                ~np.isnan(values)
            ]

            if len(available) > 0:

                prediction = float(
                    np.mean(
                        available
                    )
                )

        # ----------------------------------------------------
        # Assign initial prediction
        # ----------------------------------------------------

        if not np.isnan(
            prediction
        ):

            values[j] = prediction

    # --------------------------------------------------------
    # Online Optimization
    # --------------------------------------------------------

    for episode in range(
        episodes
    ):

        previous = values.copy()

        for j in range(
            3,
            len(values)
        ):

            # Only update originally missing cells
            if not missing_mask[j]:
                continue

            state = values[
                j-3:j
            ]

            if np.isnan(
                state
            ).any():

                continue

            prediction = float(
                np.asarray(
                    world_model.predict(
                        state.reshape(
                            1,
                            -1
                        )
                    )
                ).reshape(-1)[0]
            )

            values[j] = (
                values[j]
                + alpha
                * (
                    prediction
                    - values[j]
                )
            )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        difference = np.max(
            np.abs(
                values
                - previous
            )
        )

        if difference < 1e-6:

            break

    # --------------------------------------------------------
    # SAFETY:
    # Restore all original observed values
    # --------------------------------------------------------

    values[
        ~missing_mask
    ] = original_values[
        ~missing_mask
    ]

    return np.round(
        values,
        3
    )


# ============================================================
# FUNCTION 6
# PROCESS AFRICAN DATASET
# ============================================================

def process_african_dataset(
    df,
    world_model,
    missing_countries
):

    data = df.copy()

    # --------------------------------------------------------
    # Normalize country column
    # --------------------------------------------------------

    data["geoUnit"] = (
        data["geoUnit"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Ensure year columns exist
    # --------------------------------------------------------

    for year in AFRICA_YEARS:

        if year not in data.columns:

            data[year] = np.nan

    # --------------------------------------------------------
    # Convert year columns to numeric
    # --------------------------------------------------------

    for year in AFRICA_YEARS:

        data[year] = pd.to_numeric(
            data[year],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Impute existing countries
    # --------------------------------------------------------

    processed_rows = []

    existing_missing_count = 0

    for _, row in data.iterrows():

        original_missing = row[
            AFRICA_YEARS
        ].isna().sum()

        if original_missing > 0:

            existing_missing_count += (
                original_missing
            )

            values = impute_existing_country(
                row=row,
                world_model=world_model,
                year_cols=AFRICA_YEARS,
                alpha=0.08,
                episodes=200
            )

            for j, year in enumerate(
                AFRICA_YEARS
            ):

                row[year] = values[j]

        processed_rows.append(
            row
        )

    processed_df = pd.DataFrame(
        processed_rows
    )

    # --------------------------------------------------------
    # Generate completely missing countries
    # --------------------------------------------------------

    generated_rows = []

    for country in missing_countries:

        trajectory = (
            generate_missing_country_trajectory(
                world_model=world_model,
                reference_df=data,
                year_cols=AFRICA_YEARS,
                alpha=0.08,
                episodes=200
            )
        )

        new_row = {
            "geoUnit": country
        }

        # If dataset contains other columns,
        # initialize them as NaN.
        for col in processed_df.columns:

            if col not in new_row:

                new_row[col] = np.nan

        for j, year in enumerate(
            AFRICA_YEARS
        ):

            new_row[year] = trajectory[j]

        generated_rows.append(
            new_row
        )

    # --------------------------------------------------------
    # Add missing countries
    # --------------------------------------------------------

    if generated_rows:

        generated_df = pd.DataFrame(
            generated_rows
        )

        # Match column order
        generated_df = generated_df[
            processed_df.columns
        ]

        df_final = pd.concat(
            [
                processed_df,
                generated_df
            ],
            ignore_index=True
        )

    else:

        generated_df = pd.DataFrame(
            columns=processed_df.columns
        )

        df_final = processed_df.copy()

    # --------------------------------------------------------
    # Keep only African countries
    # --------------------------------------------------------

    df_final["geoUnit"] = (
        df_final["geoUnit"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_final = df_final[
        df_final["geoUnit"].isin(
            ALL_AFRICA
        )
    ].copy()

    # --------------------------------------------------------
    # Sort countries
    # --------------------------------------------------------

    df_final = (
        df_final
        .sort_values(
            "geoUnit"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        df_final,
        generated_df,
        existing_missing_count
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.divider()

st.header(
    "📂 Upload Excel Dataset"
)

uploaded_file = st.file_uploader(
    "Upload your Excel file",
    type=[
        "xlsx",
        "xls"
    ],
    help=(
        "Upload an Excel dataset containing "
        "a geoUnit column and annual values."
    )
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # STEP 1 — READ EXCEL
    # ========================================================

    st.subheader(
        "📥 Step 1 — Reading Uploaded Dataset"
    )

    try:

        df = pd.read_excel(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read Excel file: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    st.success(
        "Excel file uploaded successfully."
    )

    st.write(
        f"**Rows:** {len(df):,}"
    )

    st.write(
        f"**Columns:** {len(df.columns):,}"
    )


    # ========================================================
    # STEP 2 — CHECK REQUIRED COLUMNS
    # ========================================================

    st.subheader(
        "🔍 Step 2 — Dataset Validation"
    )

    if "geoUnit" not in df.columns:

        st.error(
            "The uploaded dataset must contain "
            "a 'geoUnit' column containing "
            "ISO-3 country codes."
        )

        st.stop()

    missing_year_columns = (
        validate_year_columns(
            df,
            AFRICA_YEARS
        )
    )

    if missing_year_columns:

        st.error(
            "The uploaded dataset is missing "
            "the following required year columns:"
        )

        st.write(
            missing_year_columns
        )

        st.info(
            "Required years are 2015–2025."
        )

        st.stop()

    # --------------------------------------------------------
    # Convert year columns to numeric
    # --------------------------------------------------------

    for year in AFRICA_YEARS:

        df[year] = pd.to_numeric(
            df[year],
            errors="coerce"
        )

    st.success(
        "Dataset validation successful."
    )


    # ========================================================
    # STEP 3 — AFRICAN COUNTRY COVERAGE
    # ========================================================

    st.subheader(
        "🌍 African Country Coverage"
    )

    try:

        missing_countries = (
            detect_missing_african_countries(
                df,
                "geoUnit"
            )
        )

    except ValueError as e:

        st.error(
            str(e)
        )

        st.stop()

    if missing_countries:

        st.warning(
            f"{len(missing_countries)} African "
            "countries are completely absent "
            "from the uploaded dataset."
        )

        st.write(
            "Missing African countries:"
        )

        st.dataframe(
            pd.DataFrame({
                "ISO-3": missing_countries
            }),
            use_container_width=True
        )

    else:

        st.success(
            "All African ISO-3 countries are present."
        )


    # ========================================================
    # STEP 4 — TRAIN COUNTRY WORLD MODEL
    # ========================================================

    st.subheader(
        "🧠 Step 4 — Train Country World Model"
    )

    with st.spinner(
        "Training the country-level World Model..."
    ):

        try:

            world_model, training_samples = (
                train_country_world_model(
                    df,
                    country_col="geoUnit",
                    year_cols=AFRICA_YEARS,
                    n_estimators=500
                )
            )

        except ValueError as e:

            st.error(
                f"World Model training failed: {e}"
            )

            st.stop()

    st.success(
        "Country-level World Model trained successfully."
    )

    st.info(
        f"Temporal training samples: "
        f"{training_samples:,}"
    )


    # ========================================================
    # STEP 5 — PROCESS DATA
    # ========================================================

    st.subheader(
        "🤖 Step 5 — Model-Based RL + "
        "Online Optimization"
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    status.write(
        "Processing existing African countries..."
    )

    progress.progress(
        25
    )

    try:

        (
            df_final,
            generated_df,
            existing_missing_count
        ) = process_african_dataset(
            df=df,
            world_model=world_model,
            missing_countries=missing_countries
        )

    except Exception as e:

        st.error(
            f"Imputation failed: {e}"
        )

        st.exception(e)

        st.stop()

    progress.progress(
        100
    )

    status.write(
        "Processing completed."
    )


    # ========================================================
    # STEP 6 — RESULTS
    # ========================================================

    st.subheader(
        "📊 Step 6 — Imputation Results"
    )

    st.success(
        "Model-Based RL + Online Optimization completed."
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "African countries added",
            len(missing_countries)
        )

    with col2:

        st.metric(
            "Final number of countries",
            len(df_final)
        )

    with col3:

        st.metric(
            "World-model training samples",
            training_samples
        )

    with col4:

        missing_after = (
            df_final[
                AFRICA_YEARS
            ]
            .isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing values after",
            int(missing_after)
        )


    # ========================================================
    # ORIGINAL MISSING VALUES
    # ========================================================

    missing_before = (
        df[
            AFRICA_YEARS
        ]
        .isna()
        .sum()
        .sum()
    )

    st.write(
        f"**Missing values before:** "
        f"{int(missing_before):,}"
    )

    st.write(
        f"**Missing values after:** "
        f"{int(missing_after):,}"
    )

    st.write(
        f"**Values imputed:** "
        f"{int(missing_before - missing_after):,}"
    )


    # ========================================================
    # ADDED COUNTRIES
    # ========================================================

    if len(generated_df) > 0:

        st.subheader(
            "🌍 Completely Missing African Countries Added"
        )

        added_display = generated_df[
            [
                "geoUnit"
            ] + AFRICA_YEARS
        ].copy()

        st.dataframe(
            added_display,
            use_container_width=True
        )


    # ========================================================
    # FINAL DATASET
    # ========================================================

    st.subheader(
        "📋 Final African Dataset"
    )

    st.dataframe(
        df_final,
        use_container_width=True,
        height=600
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    st.subheader(
        "✅ Final Validation"
    )

    final_countries = set(
        df_final[
            "geoUnit"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    still_missing_countries = sorted(
        ALL_AFRICA
        - final_countries
    )

    if still_missing_countries:

        st.error(
            "Some African countries are still missing:"
        )

        st.write(
            still_missing_countries
        )

    else:

        st.success(
            "All 54 African ISO-3 countries "
            "are present in the final dataset."
        )

    if missing_after == 0:

        st.success(
            "There are no missing 2015–2025 values "
            "in the final African dataset."
        )

    else:

        st.warning(
            f"{int(missing_after)} values remain missing."
        )


    # ========================================================
    # DOWNLOAD EXCEL
    # ========================================================

    st.subheader(
        "⬇️ Download Results"
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Sheet 1 — Final African Dataset
        # ----------------------------------------------------

        df_final.to_excel(
            writer,
            index=False,
            sheet_name="Imputed_Africa"
        )

        # ----------------------------------------------------
        # Sheet 2 — Added Countries
        # ----------------------------------------------------

        pd.DataFrame({
            "Missing African Countries":
                missing_countries
        }).to_excel(
            writer,
            index=False,
            sheet_name="Added_Countries"
        )

        # ----------------------------------------------------
        # Sheet 3 — Summary
        # ----------------------------------------------------

        summary_df = pd.DataFrame({
            "Metric": [
                "Original number of rows",
                "Final number of countries",
                "African countries added",
                "World model training samples",
                "Missing values before",
                "Missing values after",
                "Values imputed",
                "Year range"
            ],

            "Value": [
                len(df),
                len(df_final),
                len(missing_countries),
                training_samples,
                int(missing_before),
                int(missing_after),
                int(
                    missing_before
                    - missing_after
                ),
                "2015–2025"
            ]
        })

        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Imputation_Summary"
        )

    output.seek(0)

    st.download_button(
        label=(
            "⬇️ Download Imputed "
            "African Excel"
        ),

        data=output,

        file_name=(
            "Africa_Model_Based_RL_"
            "Online_Optimization_Imputed.xlsx"
        ),

        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    st.success(
        "Your completed African dataset is ready "
        "for download."
    )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload an Excel file above to begin."
    )

    st.markdown(
        """
        ### Required Excel structure

        Your Excel file should contain:

        **Country column**

        `geoUnit`

        **Year columns**

        `2015, 2016, 2017, ..., 2025`

        Example:

        | geoUnit | 2015 | 2016 | 2017 | ... | 2025 |
        |---|---:|---:|---:|---:|---:|
        | ZAF | 100 | 105 | 110 | ... | 150 |
        | NGA | 80 | NaN | 90 | ... | 130 |
        | KEN | 60 | 63 | NaN | ... | 100 |

        The application can handle:

        - Countries completely absent from the file
        - Countries with partially missing years
        - Multiple missing consecutive years
        - Missing values at the beginning
        - Missing values at the end

        The final output targets all **54 African countries**
        for **2015–2025**.
        """
    )
