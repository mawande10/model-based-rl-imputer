# ============================================================
# AFRICAN MODEL-BASED RL + ONLINE OPTIMIZATION IMPUTER
# Streamlit Community Cloud Application
#
# Features:
#   1. Upload Excel dataset
#   2. Detect 2015-2025 columns
#   3. Detect completely missing African countries
#   4. Train country-level temporal World Model
#   5. Impute missing years for existing countries
#   6. Generate different trajectories for completely
#      missing African countries
#   7. Apply online optimization
#   8. Display before/after results
#   9. Download final Excel workbook
# ============================================================

import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="African Model-Based RL Imputer",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# AFRICAN COUNTRY LIST — ISO-3
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
    str(y) for y in range(2015, 2026)
]


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🌍 African Model-Based RL + Online Optimization Imputer"
)

st.markdown(
    """
    This application uses a **country-level temporal World Model**
    based on Random Forest regression together with an **online
    optimization update** to populate missing values from **2015–2025**.

    It also identifies African countries that are completely absent
    from the uploaded dataset and generates a complete trajectory
    for those countries.
    """
)


# ============================================================
# HELPER — DETECT YEAR COLUMNS
# ============================================================

def detect_year_columns(df):
    """
    Detect year columns from the uploaded dataframe.

    Required period:
        2015-2025

    Returns:
        list of year columns
    """

    data = df.copy()

    data.columns = data.columns.astype(str)

    available = set(data.columns)

    year_cols = [
        year for year in AFRICA_YEARS
        if year in available
    ]

    return year_cols


# ============================================================
# STEP 1 — DETECT MISSING AFRICAN COUNTRIES
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
            f"Required country column '{country_col}' "
            "was not found."
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
# STEP 2 — CLEAN COUNTRY IDENTIFIERS
# ============================================================

def clean_country_column(
    df,
    country_col="geoUnit"
):
    """
    Standardize ISO-3 country identifiers.
    """

    data = df.copy()

    data[country_col] = (
        data[country_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return data


# ============================================================
# STEP 3 — PREPARE NUMERIC YEAR DATA
# ============================================================

def prepare_numeric_year_data(
    df,
    year_cols
):
    """
    Convert year columns to numeric.

    Strings such as:
        x
        X
        -
        blank

    become NaN and are treated as missing.
    """

    data = df.copy()

    for year in year_cols:
        data[year] = pd.to_numeric(
            data[year],
            errors="coerce"
        )

    return data


# ============================================================
# STEP 4 — TRAIN COUNTRY-LEVEL WORLD MODEL
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

    Example:

        2015, 2016, 2017
                 ↓
               2018

        2016, 2017, 2018
                 ↓
               2019

    etc.
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
        ].to_numpy(dtype=float)

        for i in range(
            3,
            len(values)
        ):

            state = values[
                i - 3:i
            ]

            target = values[i]

            # Require complete state
            if np.isnan(state).any():
                continue

            # Require complete target
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

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    )

    model.fit(
        X,
        y
    )

    return model, len(X)


# ============================================================
# STEP 5 — GLOBAL / EMPIRICAL FALLBACK
# ============================================================

def get_empirical_initial_state(
    reference_df,
    year_cols,
    country_code
):
    """
    Generate a country-specific initial 3-year state.

    IMPORTANT:
    A completely absent country has no historical observations.
    Therefore, its first three values must be estimated from
    the empirical cross-country distribution.

    To avoid identical trajectories, each ISO-3 country gets
    a deterministic seed derived from its ISO code.

    This does NOT pretend that the ISO code itself predicts
    the economic/education value. It simply provides a
    reproducible bootstrap draw from the observed distribution.
    """

    data = reference_df[
        year_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Country-specific deterministic seed
    # --------------------------------------------------------

    hash_value = hashlib.sha256(
        country_code.encode("utf-8")
    ).hexdigest()

    seed = int(
        hash_value[:8],
        16
    )

    rng = np.random.default_rng(
        seed
    )

    initial_values = []

    # --------------------------------------------------------
    # Generate 2015, 2016, 2017
    # from empirical distributions
    # --------------------------------------------------------

    for year in year_cols[:3]:

        observations = (
            data[year]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(observations) == 0:

            initial_values.append(
                np.nan
            )

            continue

        # ----------------------------------------------------
        # Empirical sampling
        # ----------------------------------------------------

        sampled = rng.choice(
            observations
        )

        # ----------------------------------------------------
        # Add a very small perturbation based on the
        # observed distribution.
        #
        # This prevents completely absent countries from
        # receiving exactly identical initial states.
        # ----------------------------------------------------

        std = np.nanstd(
            observations
        )

        if (
            np.isfinite(std)
            and std > 0
        ):

            perturbation = rng.normal(
                0,
                0.03 * std
            )

            sampled = (
                sampled
                + perturbation
            )

        # Prevent negative predictions
        sampled = max(
            0.0,
            float(sampled)
        )

        initial_values.append(
            sampled
        )

    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    global_mean = np.nanmean(
        data.to_numpy(dtype=float)
    )

    if not np.isfinite(
        global_mean
    ):
        global_mean = 0.0

    initial_values = [

        global_mean
        if not np.isfinite(value)
        else value

        for value in initial_values
    ]

    return np.asarray(
        initial_values,
        dtype=float
    )


# ============================================================
# STEP 6 — GENERATE MISSING COUNTRY TRAJECTORY
# ============================================================

def generate_missing_country_trajectory(
    world_model,
    reference_df,
    year_cols,
    country_code,
    alpha=0.08,
    episodes=200
):
    """
    Generate a complete 2015-2025 trajectory for a completely
    missing African country.

    2015-2017:
        Empirical country-specific bootstrap.

    2018-2025:
        World Model prediction.

    Then:
        Online Optimization

    The country-specific bootstrap prevents all absent
    African countries from receiving the same trajectory.
    """

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_values = get_empirical_initial_state(
        reference_df=reference_df,
        year_cols=year_cols,
        country_code=country_code
    )

    # First three years
    values[:3] = initial_values

    # --------------------------------------------------------
    # MODEL-BASED TRAJECTORY
    # --------------------------------------------------------

    for j in range(
        3,
        len(year_cols)
    ):

        state = values[
            j - 3:j
        ]

        if np.isnan(
            state
        ).any():

            # Robust fallback
            finite_values = values[
                np.isfinite(values)
            ]

            if len(finite_values) > 0:
                prediction = float(
                    np.mean(
                        finite_values
                    )
                )
            else:
                prediction = 0.0

        else:

            prediction = world_model.predict(
                state.reshape(1, -1)
            )

            prediction = float(
                np.asarray(
                    prediction
                ).reshape(-1)[0]
            )

        # ----------------------------------------------------
        # Prevent negative values
        # ----------------------------------------------------

        values[j] = max(
            0.0,
            prediction
        )

    # --------------------------------------------------------
    # ONLINE OPTIMIZATION
    # --------------------------------------------------------

    for _ in range(
        episodes
    ):

        previous = values.copy()

        for j in range(
            3,
            len(values)
        ):

            state = values[
                j - 3:j
            ]

            if np.isnan(
                state
            ).any():

                continue

            prediction = world_model.predict(
                state.reshape(1, -1)
            )

            prediction = float(
                np.asarray(
                    prediction
                ).reshape(-1)[0]
            )

            # ------------------------------------------------
            # Online update
            # ------------------------------------------------

            values[j] = (
                values[j]
                + alpha
                * (
                    prediction
                    - values[j]
                )
            )

            values[j] = max(
                0.0,
                values[j]
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
# STEP 7 — IMPUTE EXISTING COUNTRIES
# ============================================================

def impute_existing_country_missing_values(
    df,
    world_model,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Impute missing values for countries already present
    in the uploaded dataset.

    IMPORTANT:
        Original observed values are NEVER modified.

    Missing values are estimated using:

        1. World Model
        2. Nearest available neighbours
        3. Row mean
        4. Global mean

    Then online optimization is applied.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for year in year_cols:

        result[year] = pd.to_numeric(
            result[year],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remember original missing positions
    # --------------------------------------------------------

    missing_mask = (
        result[year_cols]
        .isna()
        .copy()
    )

    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    numeric_matrix = (
        result[year_cols]
        .to_numpy(dtype=float)
    )

    global_mean = np.nanmean(
        numeric_matrix
    )

    if not np.isfinite(
        global_mean
    ):
        global_mean = 0.0

    # ========================================================
    # PROCESS EACH COUNTRY
    # ========================================================

    for idx in result.index:

        values = (
            result.loc[
                idx,
                year_cols
            ]
            .to_numpy(dtype=float)
        )

        original_mask = (
            missing_mask.loc[
                idx,
                year_cols
            ]
            .to_numpy(dtype=bool)
        )

        # ----------------------------------------------------
        # If no missing values, do nothing
        # ----------------------------------------------------

        if not original_mask.any():
            continue

        # ====================================================
        # ITERATIVE MODEL-BASED IMPUTATION
        # ====================================================

        for _ in range(
            episodes
        ):

            previous = values.copy()

            # ------------------------------------------------
            # Process ONLY originally missing values
            # ------------------------------------------------

            for j in range(
                len(values)
            ):

                if not original_mask[j]:
                    continue

                prediction = np.nan

                # ============================================
                # 1. WORLD MODEL
                # ============================================

                if j >= 3:

                    state = values[
                        j - 3:j
                    ]

                    if not np.isnan(
                        state
                    ).any():

                        prediction_array = (
                            world_model.predict(
                                state.reshape(1, -1)
                            )
                        )

                        prediction = float(
                            np.asarray(
                                prediction_array
                            ).reshape(-1)[0]
                        )

                # ============================================
                # 2. NEAREST LEFT / RIGHT VALUES
                # ============================================

                if not np.isfinite(
                    prediction
                ):

                    left = None
                    right = None

                    # ----------------------------------------
                    # Find previous available value
                    # ----------------------------------------

                    for k in range(
                        j - 1,
                        -1,
                        -1
                    ):

                        if np.isfinite(
                            values[k]
                        ):

                            left = float(
                                values[k]
                            )

                            break

                    # ----------------------------------------
                    # Find next available value
                    # ----------------------------------------

                    for k in range(
                        j + 1,
                        len(values)
                    ):

                        if np.isfinite(
                            values[k]
                        ):

                            right = float(
                                values[k]
                            )

                            break

                    # ----------------------------------------
                    # Both neighbours
                    # ----------------------------------------

                    if (
                        left is not None
                        and right is not None
                    ):

                        prediction = (
                            left
                            + right
                        ) / 2.0

                    # ----------------------------------------
                    # Only left
                    # ----------------------------------------

                    elif left is not None:

                        prediction = left

                    # ----------------------------------------
                    # Only right
                    # ----------------------------------------

                    elif right is not None:

                        prediction = right

                # ============================================
                # 3. ROW MEAN
                # ============================================

                if not np.isfinite(
                    prediction
                ):

                    finite_values = values[
                        np.isfinite(values)
                    ]

                    if len(
                        finite_values
                    ) > 0:

                        prediction = float(
                            np.mean(
                                finite_values
                            )
                        )

                # ============================================
                # 4. GLOBAL MEAN
                # ============================================

                if not np.isfinite(
                    prediction
                ):

                    prediction = global_mean

                # ============================================
                # ONLINE OPTIMIZATION
                # ============================================

                if not np.isfinite(
                    values[j]
                ):

                    # First RL/model estimate
                    values[j] = prediction

                else:

                    # Online optimization update
                    values[j] = (
                        values[j]
                        + alpha
                        * (
                            prediction
                            - values[j]
                        )
                    )

                # Prevent invalid negative predictions
                values[j] = max(
                    0.0,
                    float(values[j])
                )

            # ------------------------------------------------
            # Check convergence
            # ------------------------------------------------

            difference = np.nanmax(
                np.abs(
                    values
                    - previous
                )
            )

            if (
                not np.isfinite(
                    difference
                )
                or difference < 1e-6
            ):

                break

        # ----------------------------------------------------
        # Save country result
        # ----------------------------------------------------

        result.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

    return result


# ============================================================
# STEP 8 — ADD COMPLETELY MISSING AFRICAN COUNTRIES
# ============================================================

def add_missing_african_countries(
    df,
    missing_countries,
    world_model,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Create rows for African countries that are completely
    absent from the uploaded dataset.
    """

    result = df.copy()

    added_rows = []

    for country in missing_countries:

        trajectory = (
            generate_missing_country_trajectory(
                world_model=world_model,
                reference_df=df,
                year_cols=year_cols,
                country_code=country,
                alpha=alpha,
                episodes=episodes
            )
        )

        row = {
            "geoUnit": country
        }

        for i, year in enumerate(
            year_cols
        ):

            row[year] = trajectory[i]

        added_rows.append(
            row
        )

    # --------------------------------------------------------
    # Append rows
    # --------------------------------------------------------

    if added_rows:

        added_df = pd.DataFrame(
            added_rows
        )

        # Make sure column order follows original dataset
        for col in result.columns:

            if col not in added_df.columns:

                added_df[col] = np.nan

        added_df = added_df[
            result.columns
        ]

        result = pd.concat(
            [
                result,
                added_df
            ],
            ignore_index=True
        )

    return result


# ============================================================
# STEP 9 — VALIDATION
# ============================================================

def calculate_validation_statistics(
    original_df,
    final_df,
    year_cols
):
    """
    Calculate simple before/after missing-value statistics.
    """

    original_numeric = (
        original_df[
            year_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    final_numeric = (
        final_df[
            year_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    before_missing = int(
        original_numeric.isna()
        .sum()
        .sum()
    )

    after_missing = int(
        final_numeric.isna()
        .sum()
        .sum()
    )

    return (
        before_missing,
        after_missing
    )


# ============================================================
# STEP 10 — EXCEL EXPORT
# ============================================================

def create_excel_download(
    df_final,
    missing_countries,
    original_missing,
    year_cols
):
    """
    Create a multi-sheet Excel workbook.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Final data
        # ----------------------------------------------------

        df_final.to_excel(
            writer,
            index=False,
            sheet_name="Imputed_Africa"
        )

        # ----------------------------------------------------
        # Added countries
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
        # Original missing values
        # ----------------------------------------------------

        pd.DataFrame({
            "Metric": [
                "Original missing values",
                "Final missing values"
            ],
            "Count": [
                original_missing,
                int(
                    df_final[
                        year_cols
                    ]
                    .isna()
                    .sum()
                    .sum()
                )
            ]
        }).to_excel(
            writer,
            index=False,
            sheet_name="Validation"
        )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Settings"
)

alpha = st.sidebar.slider(
    "Online optimization α",
    min_value=0.01,
    max_value=0.50,
    value=0.08,
    step=0.01
)

episodes = st.sidebar.slider(
    "Optimization iterations",
    min_value=10,
    max_value=500,
    value=200,
    step=10
)

n_estimators = st.sidebar.slider(
    "Random Forest trees",
    min_value=100,
    max_value=1000,
    value=500,
    step=100
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "📂 Upload Excel Dataset"
)

uploaded_file = st.file_uploader(
    "Upload your Excel file",
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
    # READ EXCEL FILE
    # ========================================================

    try:

        df = pd.read_excel(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read the Excel file: {e}"
        )

        st.stop()

    # ========================================================
    # STANDARDIZE COLUMN NAMES
    # ========================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # CHECK COUNTRY COLUMN
    # ========================================================

    if "geoUnit" not in df.columns:

        st.error(
            "The uploaded dataset must contain "
            "a 'geoUnit' column."
        )

        st.write(
            "Detected columns:"
        )

        st.write(
            df.columns.tolist()
        )

        st.stop()

    # ========================================================
    # DETECT YEAR COLUMNS
    # ========================================================

    year_cols = detect_year_columns(
        df
    )

    missing_required_years = sorted(
        set(AFRICA_YEARS)
        - set(year_cols),
        key=int
    )

    if missing_required_years:

        st.error(
            "The uploaded dataset is missing "
            "the following required year columns:"
        )

        st.write(
            missing_required_years
        )

        st.stop()

    # ========================================================
    # CLEAN DATA
    # ========================================================

    df = clean_country_column(
        df,
        "geoUnit"
    )

    df = prepare_numeric_year_data(
        df,
        year_cols
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.header(
        "📊 Dataset Information"
    )

    col1, col2, col3, col4 = st.columns(4)

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
            "Year range",
            "2015–2025"
        )

    with col4:

        st.metric(
            "Countries",
            df["geoUnit"].nunique()
        )

    # ========================================================
    # SHOW ORIGINAL DATA
    # ========================================================

    with st.expander(
        "👁️ View Uploaded Dataset"
    ):

        st.dataframe(
            df,
            use_container_width=True
        )

    # ========================================================
    # STEP 1 — AFRICAN COUNTRY COVERAGE
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
            f"{len(missing_countries)} African "
            "countries are completely absent "
            "from the uploaded dataset."
        )

        st.write(
            "Missing African countries:"
        )

        st.dataframe(
            pd.DataFrame({
                "ISO-3":
                    missing_countries
            }),
            use_container_width=True
        )

    else:

        st.success(
            "All African ISO-3 countries "
            "are present."
        )

    # ========================================================
    # ORIGINAL MISSING VALUES
    # ========================================================

    original_missing = int(
        df[
            year_cols
        ]
        .isna()
        .sum()
        .sum()
    )

    st.info(
        f"Original missing year values: "
        f"{original_missing:,}"
    )

    # ========================================================
    # STEP 4 — TRAIN COUNTRY WORLD MODEL
    # ========================================================

    st.subheader(
        "🧠 Country-Level World Model"
    )

    try:

        world_model, training_samples = (
            train_country_world_model(
                df,
                country_col="geoUnit",
                year_cols=AFRICA_YEARS,
                n_estimators=n_estimators
            )
        )

        st.success(
            "Country-level World Model "
            "trained successfully."
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
    # RUN BUTTON
    # ========================================================

    st.divider()

    run_model = st.button(
        "🚀 Run Model-Based RL + Online Optimization",
        type="primary",
        use_container_width=True
    )

    if run_model:

        # ====================================================
        # STEP 5 — IMPUTE EXISTING COUNTRIES
        # ====================================================

        st.header(
            "🔄 Imputing Missing Years"
        )

        progress = st.progress(
            0
        )

        status = st.empty()

        status.info(
            "Step 1/2: Imputing missing years "
            "for countries already present..."
        )

        df_existing_imputed = (
            impute_existing_country_missing_values(
                df=df,
                world_model=world_model,
                year_cols=year_cols,
                alpha=alpha,
                episodes=episodes
            )
        )

        progress.progress(
            50
        )

        # ====================================================
        # STEP 6 — ADD COMPLETELY MISSING COUNTRIES
        # ====================================================

        status.info(
            "Step 2/2: Generating trajectories "
            "for completely missing African countries..."
        )

        df_final = (
            add_missing_african_countries(
                df=df_existing_imputed,
                missing_countries=missing_countries,
                world_model=world_model,
                year_cols=year_cols,
                alpha=alpha,
                episodes=episodes
            )
        )

        progress.progress(
            100
        )

        status.success(
            "Model-Based RL + Online Optimization "
            "completed."
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        before_missing, after_missing = (
            calculate_validation_statistics(
                original_df=df,
                final_df=df_final,
                year_cols=year_cols
            )
        )

        # ====================================================
        # RESULTS
        # ====================================================

        st.header(
            "📈 Imputation Results"
        )

        col1, col2, col3, col4 = st.columns(4)

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

            st.metric(
                "Remaining missing values",
                after_missing
            )

        # ====================================================
        # VALIDATION STATUS
        # ====================================================

        if after_missing == 0:

            st.success(
                "✅ All 2015–2025 missing values "
                "have been populated."
            )

        else:

            st.warning(
                f"{after_missing:,} missing values "
                "remain after processing."
            )

        # ====================================================
        # BEFORE / AFTER
        # ====================================================

        st.subheader(
            "📊 Before vs After"
        )

        comparison_df = pd.DataFrame({
            "Metric": [
                "Missing values",
                "Number of countries"
            ],
            "Before": [
                before_missing,
                df["geoUnit"].nunique()
            ],
            "After": [
                after_missing,
                df_final["geoUnit"].nunique()
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # SHOW ADDED AFRICAN COUNTRIES
        # ====================================================

        if missing_countries:

            st.subheader(
                "🌍 Newly Added African Countries"
            )

            added_only = df_final[
                df_final[
                    "geoUnit"
                ].isin(
                    missing_countries
                )
            ].copy()

            st.dataframe(
                added_only[
                    [
                        "geoUnit"
                    ]
                    + year_cols
                ],
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Check whether trajectories are identical
            # ------------------------------------------------

            if len(added_only) > 1:

                trajectory_matrix = (
                    added_only[
                        year_cols
                    ]
                    .to_numpy()
                )

                unique_trajectories = (
                    np.unique(
                        trajectory_matrix,
                        axis=0
                    ).shape[0]
                )

                st.info(
                    f"Generated "
                    f"{unique_trajectories} unique "
                    f"trajectory/trajectories for "
                    f"{len(added_only)} added countries."
                )

        # ====================================================
        # FULL FINAL DATASET
        # ====================================================

        st.subheader(
            "📋 Final Imputed Dataset"
        )

        st.dataframe(
            df_final,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD EXCEL
        # ====================================================

        output = create_excel_download(
            df_final=df_final,
            missing_countries=missing_countries,
            original_missing=original_missing,
            year_cols=year_cols
        )

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
            ),
            use_container_width=True
        )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander(
            "🧠 View Model Methodology"
        ):

            st.markdown(
                """
                ### Model-Based RL + Online Optimization

                The temporal state is:

                **Sₜ = [yₜ₋₃, yₜ₋₂, yₜ₋₁]**

                The Random Forest World Model estimates:

                **ŷₜ = f(Sₜ)**

                The online optimization update is:

                **yₜ ← yₜ + α(ŷₜ − yₜ)**

                where:

                - **α** = online optimization learning rate
                - **ŷₜ** = World Model prediction
                - **yₜ** = current estimated value

                ### Existing countries

                Observed values are preserved.

                Only values that were originally missing
                are changed.

                ### Completely missing African countries

                Since a completely absent country has no historical
                observations, its initial 2015–2017 state is generated
                from the empirical distribution of the uploaded
                countries.

                A deterministic ISO-3-specific seed produces
                different bootstrap states for different countries.

                The World Model then generates the remaining trajectory
                from 2018–2025.

                ### Important

                The trajectory for a completely absent country is an
                **model-based estimate**, not an observed historical
                value.
                """
            )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload an Excel file to begin."
    )

    st.markdown(
        """
        ### Required Excel structure

        Your Excel file should contain:

        | geoUnit | 2015 | 2016 | 2017 | ... | 2025 |
        |---|---:|---:|---:|---:|---:|
        | AGO | ... | ... | ... | ... | ... |
        | BDI | ... | ... | ... | ... | ... |
        | ZAF | ... | ... | ... | ... | ... |

        **`geoUnit` must contain ISO-3 country codes.**

        Missing values can be blank, `x`, `X`, or other
        non-numeric entries because they will be converted
        to `NaN`.
        """
    )
