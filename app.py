# ============================================================
# AFRICA MODEL-BASED RL + ONLINE OPTIMIZATION
# STREAMLIT APPLICATION
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
    page_title="Africa Model-Based RL Imputer",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# AFRICAN COUNTRIES — ISO-3
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


# ============================================================
# AFRICAN YEARS
# ============================================================

AFRICA_YEARS = [
    str(y)
    for y in range(2015, 2026)
]


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🌍 African Data Imputation"
)

st.subheader(
    "Model-Based RL + Online Optimization"
)

st.write(
    """
    Upload an Excel dataset containing African country data.
    
    The application will:
    
    1. Detect missing African countries.
    2. Detect missing values for 2015–2025.
    3. Train a cross-country temporal World Model.
    4. Generate country-specific trajectories.
    5. Apply Model-Based RL + Online Optimization.
    6. Preserve all observed values.
    7. Add completely missing African countries.
    8. Produce an imputed Excel file for download.
    """
)


# ============================================================
# FUNCTION 1
# DETECT MISSING AFRICAN COUNTRIES
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
# DETECT YEAR COLUMNS
# ============================================================

def detect_year_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    year_cols = [
        year
        for year in AFRICA_YEARS
        if year in df.columns
    ]

    return year_cols


# ============================================================
# FUNCTION 3
# PREPARE DATA
# ============================================================

def prepare_numeric_data(
    df,
    year_cols
):

    data = df.copy()

    for col in year_cols:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data


# ============================================================
# FUNCTION 4
# TRAIN COUNTRY-LEVEL WORLD MODEL
# ============================================================

def train_country_world_model(
    df,
    country_col="geoUnit",
    year_cols=None,
    n_estimators=500
):
    """
    Train a temporal Random Forest world model.

    State:
        [Y(t-3), Y(t-2), Y(t-1)]

    Target:
        Y(t)
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    data = df.copy()

    # ------------------------------------------
    # Ensure numeric values
    # ------------------------------------------

    for col in year_cols:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # ------------------------------------------
    # Create temporal training samples
    # ------------------------------------------

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

    if len(X) < 5:

        raise ValueError(
            "Not enough complete temporal observations "
            "to train the world model."
        )

    # ------------------------------------------
    # Random Forest World Model
    # ------------------------------------------

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
# FUNCTION 5
# GENERATE MISSING COUNTRY TRAJECTORY
# ============================================================

def generate_missing_country_trajectory(
    world_model,
    reference_df,
    year_cols,
    country_code,
    alpha=0.08,
    episodes=200,
    random_state=42
):
    """
    Generate a country-specific 2015–2025 trajectory
    for an African country that is completely absent
    from the uploaded dataset.

    Country-specific initialization and controlled
    stochastic variation prevent all missing countries
    from receiving identical trajectories.
    """

    # ------------------------------------------
    # Country-specific random generator
    # ------------------------------------------

    rng = np.random.default_rng(
        random_state
        +
        sum(
            ord(c)
            for c in str(country_code)
        )
    )

    # ------------------------------------------
    # Prepare reference data
    # ------------------------------------------

    data = reference_df[
        year_cols
    ].copy()

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    matrix = data.to_numpy(
        dtype=float
    )

    # ------------------------------------------
    # Global statistics
    # ------------------------------------------

    global_mean = np.nanmean(
        matrix
    )

    global_std = np.nanstd(
        matrix
    )

    if np.isnan(global_mean):

        global_mean = 0.0

    if (
        np.isnan(global_std)
        or global_std == 0
    ):

        global_std = max(
            abs(global_mean) * 0.05,
            1e-6
        )

    # ------------------------------------------
    # Year-specific statistics
    # ------------------------------------------

    yearly_medians = []
    yearly_means = []
    yearly_stds = []

    for year in year_cols:

        values = data[
            year
        ].dropna().to_numpy(
            dtype=float
        )

        if len(values) == 0:

            yearly_medians.append(
                global_mean
            )

            yearly_means.append(
                global_mean
            )

            yearly_stds.append(
                global_std
            )

        else:

            yearly_medians.append(
                np.median(values)
            )

            yearly_means.append(
                np.mean(values)
            )

            std = np.std(values)

            if (
                np.isnan(std)
                or std == 0
            ):

                std = global_std

            yearly_stds.append(
                std
            )

    yearly_medians = np.asarray(
        yearly_medians,
        dtype=float
    )

    yearly_means = np.asarray(
        yearly_means,
        dtype=float
    )

    yearly_stds = np.asarray(
        yearly_stds,
        dtype=float
    )

    # ------------------------------------------
    # Country-specific profile
    # ------------------------------------------

    country_factor = rng.normal(
        loc=1.0,
        scale=0.12
    )

    country_factor = np.clip(
        country_factor,
        0.75,
        1.25
    )

    # ------------------------------------------
    # Initial 3-year state
    # ------------------------------------------

    initial_values = []

    for j in range(3):

        base_value = (
            0.60
            * yearly_medians[j]
            +
            0.40
            * yearly_means[j]
        )

        variation = rng.normal(
            loc=0.0,
            scale=yearly_stds[j] * 0.20
        )

        value = (
            base_value
            * country_factor
            +
            variation
        )

        value = max(
            value,
            0.0
        )

        initial_values.append(
            value
        )

    initial_values = np.asarray(
        initial_values,
        dtype=float
    )

    # ------------------------------------------
    # Create trajectory
    # ------------------------------------------

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    values[:3] = initial_values

    # ------------------------------------------
    # Model-Based RL
    # ------------------------------------------

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
            state.reshape(1, -1)
        )

        prediction = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        # Country-specific uncertainty
        noise = rng.normal(
            loc=0.0,
            scale=yearly_stds[j] * 0.05
        )

        prediction += noise

        prediction = max(
            prediction,
            0.0
        )

        values[j] = prediction

    # ------------------------------------------
    # Online Optimization
    # ------------------------------------------

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

            if np.isnan(state).any():
                continue

            prediction = world_model.predict(
                state.reshape(1, -1)
            )

            prediction = float(
                np.asarray(
                    prediction
                ).reshape(-1)[0]
            )

            # Temporal consistency
            temporal_target = np.mean(
                state
            )

            optimized_target = (
                0.75 * prediction
                +
                0.25 * temporal_target
            )

            values[j] = (
                values[j]
                +
                alpha
                *
                (
                    optimized_target
                    -
                    values[j]
                )
            )

            values[j] = max(
                values[j],
                0.0
            )

        difference = np.max(
            np.abs(
                values - previous
            )
        )

        if difference < 1e-6:
            break

    # ------------------------------------------
    # Country-specific trend
    # ------------------------------------------

    trend_strength = rng.normal(
        loc=0.0,
        scale=0.003
    )

    for j in range(
        3,
        len(values)
    ):

        years_from_start = (
            j - 2
        )

        values[j] *= (
            1
            +
            trend_strength
            *
            years_from_start
        )

    # ------------------------------------------
    # Final cleanup
    # ------------------------------------------

    values = np.nan_to_num(
        values,
        nan=global_mean,
        posinf=global_mean,
        neginf=0.0
    )

    values = np.maximum(
        values,
        0.0
    )

    return np.round(
        values,
        3
    )


# ============================================================
# FUNCTION 6
# IMPUTE MISSING YEARS FOR EXISTING COUNTRIES
# ============================================================

def impute_existing_country_missing_values(
    df,
    world_model,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Fill missing values for countries that already
    exist in the uploaded dataset.

    Observed values are NEVER modified.
    """

    result = df.copy()

    # ------------------------------------------
    # Convert years to numeric
    # ------------------------------------------

    for col in year_cols:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    # Original missing-value mask
    missing_mask = result[
        year_cols
    ].isna()

    # ------------------------------------------
    # Process each country
    # ------------------------------------------

    for idx in range(
        len(result)
    ):

        values = (
            result
            .loc[idx, year_cols]
            .astype(float)
            .to_numpy()
        )

        original_missing = (
            missing_mask
            .loc[idx, year_cols]
            .to_numpy()
        )

        # --------------------------------------
        # Multiple optimization episodes
        # --------------------------------------

        for episode in range(
            episodes
        ):

            previous = values.copy()

            # ----------------------------------
            # Update only originally missing
            # values
            # ----------------------------------

            for j in range(
                len(values)
            ):

                if not original_missing[j]:
                    continue

                prediction = np.nan

                # ==================================
                # 1. World Model
                # ==================================

                if j >= 3:

                    state = values[
                        j-3:j
                    ]

                    if not np.isnan(
                        state
                    ).any():

                        prediction = (
                            world_model
                            .predict(
                                state.reshape(
                                    1, -1
                                )
                            )[0]
                        )

                # ==================================
                # 2. Neighbour information
                # ==================================

                if np.isnan(
                    prediction
                ):

                    neighbours = []

                    # Previous
                    if j > 0:

                        if not np.isnan(
                            values[j-1]
                        ):

                            neighbours.append(
                                values[j-1]
                            )

                    # Next
                    if j < len(values)-1:

                        if not np.isnan(
                            values[j+1]
                        ):

                            neighbours.append(
                                values[j+1]
                            )

                    if len(
                        neighbours
                    ) == 2:

                        prediction = (
                            np.mean(
                                neighbours
                            )
                        )

                    elif len(
                        neighbours
                    ) == 1:

                        prediction = (
                            neighbours[0]
                        )

                # ==================================
                # 3. Nearest available value
                # ==================================

                if np.isnan(
                    prediction
                ):

                    available = (
                        values[
                            ~np.isnan(values)
                        ]
                    )

                    if len(
                        available
                    ) > 0:

                        prediction = (
                            np.median(
                                available
                            )
                        )

                # ==================================
                # 4. Global fallback
                # ==================================

                if np.isnan(
                    prediction
                ):

                    all_values = (
                        result[
                            year_cols
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    prediction = (
                        np.nanmedian(
                            all_values
                        )
                    )

                # ==================================
                # Insert prediction
                # ==================================

                if np.isnan(
                    values[j]
                ):

                    values[j] = (
                        prediction
                    )

                else:

                    values[j] = (
                        values[j]
                        +
                        alpha
                        *
                        (
                            prediction
                            -
                            values[j]
                        )
                    )

            # ----------------------------------
            # Check convergence
            # ----------------------------------

            if np.all(
                ~np.isnan(values)
            ):

                break

            difference = np.nanmax(
                np.abs(
                    values - previous
                )
            )

            if difference < 1e-6:

                break

        result.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

    return result


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your Excel file",
    type=[
        "xlsx",
        "xls"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # STEP 1 — READ EXCEL
    # ========================================================

    try:

        df = pd.read_excel(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read Excel file: {e}"
        )

        st.stop()

    # ------------------------------------------
    # Normalize column names
    # ------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # BASIC DATASET INFORMATION
    # ========================================================

    st.header(
        "📊 Uploaded Dataset"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            len(df)
        )

    with col2:

        st.metric(
            "Columns",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Missing Cells",
            int(
                df.isna()
                .sum()
                .sum()
            )
        )

    # ========================================================
    # CHECK COUNTRY COLUMN
    # ========================================================

    if "geoUnit" not in df.columns:

        st.error(
            """
            The uploaded Excel file must contain a
            `geoUnit` column containing ISO-3 country codes.
            """
        )

        st.stop()

    # ========================================================
    # CHECK YEAR COLUMNS
    # ========================================================

    year_cols = detect_year_columns(
        df
    )

    missing_year_columns = [
        year
        for year in AFRICA_YEARS
        if year not in df.columns
    ]

    if missing_year_columns:

        st.error(
            "The following required year columns "
            "are missing from the uploaded file:"
        )

        st.write(
            missing_year_columns
        )

        st.info(
            "The application requires annual columns "
            "from 2015 through 2025."
        )

        st.stop()

    st.success(
        "All required year columns 2015–2025 were found."
    )

    # ========================================================
    # PREPARE NUMERIC DATA
    # ========================================================

    df = prepare_numeric_data(
        df,
        AFRICA_YEARS
    )

    # ========================================================
    # STEP 2 — AFRICAN COUNTRY COVERAGE
    # ========================================================

    st.subheader(
        "🌍 African Country Coverage"
    )

    missing_countries = (
        detect_missing_african_countries(
            df,
            "geoUnit"
        )
    )

    if missing_countries:

        st.warning(
            f"{len(missing_countries)} African countries "
            "are completely absent from the uploaded dataset."
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
    # STEP 3 — ORIGINAL MISSING VALUES
    # ========================================================

    st.subheader(
        "🔎 Missing Data Before Imputation"
    )

    original_missing_total = int(
        df[
            AFRICA_YEARS
        ]
        .isna()
        .sum()
        .sum()
    )

    countries_before = len(
        df
    )

    st.metric(
        "Missing year values",
        original_missing_total
    )

    st.metric(
        "Countries in uploaded dataset",
        countries_before
    )

    # ========================================================
    # STEP 4 — TRAIN COUNTRY WORLD MODEL
    # ========================================================

    st.subheader(
        "🧠 Training Country-Level World Model"
    )

    try:

        world_model, training_samples = (
            train_country_world_model(
                df,
                country_col="geoUnit",
                year_cols=AFRICA_YEARS,
                n_estimators=500
            )
        )

        st.success(
            "Country-level World Model trained successfully."
        )

        st.info(
            f"Temporal training samples: "
            f"{training_samples:,}"
        )

    except ValueError as e:

        st.error(
            f"World Model training failed: {e}"
        )

        st.stop()

    # ========================================================
    # STEP 5 — IMPUTE MISSING VALUES IN EXISTING COUNTRIES
    # ========================================================

    st.subheader(
        "🔄 Imputing Missing Years"
    )

    with st.spinner(
        "Running Model-Based RL + Online Optimization..."
    ):

        df_imputed = (
            impute_existing_country_missing_values(
                df,
                world_model,
                AFRICA_YEARS,
                alpha=0.08,
                episodes=200
            )
        )

    st.success(
        "Missing values in existing African countries "
        "have been imputed."
    )

    # ========================================================
    # STEP 6 — ADD COMPLETELY MISSING AFRICAN COUNTRIES
    # ========================================================

    st.subheader(
        "🌍 Generating Completely Missing African Countries"
    )

    new_country_rows = []

    for country_code in missing_countries:

        trajectory = (
            generate_missing_country_trajectory(
                world_model=world_model,
                reference_df=df_imputed,
                year_cols=AFRICA_YEARS,
                country_code=country_code,
                alpha=0.08,
                episodes=200,
                random_state=42
            )
        )

        country_row = {
            "geoUnit": country_code
        }

        for year, value in zip(
            AFRICA_YEARS,
            trajectory
        ):

            country_row[year] = value

        new_country_rows.append(
            country_row
        )

    # ========================================================
    # CREATE MISSING COUNTRY TABLE
    # ========================================================

    if new_country_rows:

        missing_table = pd.DataFrame(
            new_country_rows
        )

        missing_table = missing_table[
            ["geoUnit"]
            +
            AFRICA_YEARS
        ]

    else:

        missing_table = pd.DataFrame(
            columns=[
                "geoUnit"
            ]
            +
            AFRICA_YEARS
        )

    # ========================================================
    # APPEND MISSING COUNTRIES
    # ========================================================

    if len(
        missing_table
    ) > 0:

        df_final = pd.concat(
            [
                df_imputed,
                missing_table
            ],
            ignore_index=True
        )

    else:

        df_final = df_imputed.copy()

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    other_columns = [
        col
        for col in df_final.columns
        if col not in AFRICA_YEARS
        and col != "geoUnit"
    ]

    df_final = df_final[
        ["geoUnit"]
        +
        AFRICA_YEARS
        +
        other_columns
    ]

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    for col in AFRICA_YEARS:

        df_final[col] = pd.to_numeric(
            df_final[col],
            errors="coerce"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "✅ Imputation Results"
    )

    final_missing = int(
        df_final[
            AFRICA_YEARS
        ]
        .isna()
        .sum()
        .sum()
    )

    countries_after = len(
        df_final
    )

    # ========================================================
    # RESULT METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "African countries added",
            len(
                missing_countries
            )
        )

    with col2:

        st.metric(
            "Final number of countries",
            countries_after
        )

    with col3:

        st.metric(
            "World-model training samples",
            training_samples
        )

    with col4:

        st.metric(
            "Remaining missing values",
            final_missing
        )

    # ========================================================
    # STATUS
    # ========================================================

    if final_missing == 0:

        st.success(
            "🎉 All African country-year values "
            "for 2015–2025 are populated."
        )

    else:

        st.warning(
            f"{final_missing} missing values remain."
        )

    # ========================================================
    # SHOW GENERATED COUNTRIES
    # ========================================================

    if len(
        missing_table
    ) > 0:

        st.subheader(
            "➕ Newly Added African Countries"
        )

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    # ========================================================
    # CHECK THAT GENERATED COUNTRIES ARE DIFFERENT
    # ========================================================

    if len(
        missing_table
    ) > 1:

        unique_trajectories = (
            missing_table[
                AFRICA_YEARS
            ]
            .drop_duplicates()
            .shape[0]
        )

        if (
            unique_trajectories
            <
            len(missing_table)
        ):

            st.warning(
                "Some generated country trajectories "
                "are identical. This can occur when the "
                "reference data contain very limited variation."
            )

        else:

            st.success(
                "Country-specific trajectories were generated "
                "for the newly added African countries."
            )

    # ========================================================
    # FINAL DATASET
    # ========================================================

    st.subheader(
        "📋 Final Imputed Dataset"
    )

    st.dataframe(
        df_final,
        use_container_width=True,
        height=600
    )

    # ========================================================
    # BEFORE / AFTER COMPARISON
    # ========================================================

    st.subheader(
        "📈 Before vs After"
    )

    comparison = pd.DataFrame({
        "Measure": [
            "Countries before",
            "Countries after",
            "Missing values before",
            "Missing values after",
            "African countries added",
            "World-model training samples"
        ],
        "Before / Original": [
            countries_before,
            "",
            original_missing_total,
            "",
            "",
            ""
        ],
        "After / Final": [
            "",
            countries_after,
            "",
            final_missing,
            len(missing_countries),
            training_samples
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DOWNLOAD EXCEL
    # ========================================================

    st.subheader(
        "📥 Download Results"
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # --------------------------------------
        # Final imputed dataset
        # --------------------------------------

        df_final.to_excel(
            writer,
            index=False,
            sheet_name="Imputed_Africa"
        )

        # --------------------------------------
        # Added countries
        # --------------------------------------

        pd.DataFrame({
            "Missing African Countries":
                missing_countries
        }).to_excel(
            writer,
            index=False,
            sheet_name="Added_Countries"
        )

        # --------------------------------------
        # Summary
        # --------------------------------------

        pd.DataFrame({
            "Metric": [
                "Countries before",
                "Countries after",
                "Missing values before",
                "Missing values after",
                "African countries added",
                "World-model training samples"
            ],
            "Value": [
                countries_before,
                countries_after,
                original_missing_total,
                final_missing,
                len(missing_countries),
                training_samples
            ]
        }).to_excel(
            writer,
            index=False,
            sheet_name="Summary"
        )

    output.seek(0)

    # ========================================================
    # DOWNLOAD BUTTON
    # ========================================================

    st.download_button(
        label=(
            "⬇️ Download Imputed African Excel"
        ),
        data=output.getvalue(),
        file_name=(
            "Africa_Model_Based_RL_"
            "Online_Optimization_Imputed.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

else:

    # ========================================================
    # NO FILE UPLOADED
    # ========================================================

    st.info(
        """
        👆 Please upload an Excel file to begin.

        Required structure:

        `geoUnit | 2015 | 2016 | ... | 2025`

        The `geoUnit` column should contain ISO-3
        country codes.
        """
    )

    st.markdown(
        """
        ### Supported process

        **Upload Excel**
        ↓  
        **Check African countries**
        ↓  
        **Detect missing countries**
        ↓  
        **Train World Model**
        ↓  
        **Model-Based RL**
        ↓  
        **Online Optimization**
        ↓  
        **Generate missing countries**
        ↓  
        **Fill missing years**
        ↓  
        **Download Excel**
        """
    )
