# ==========================================
# MODEL-BASED RL + ONLINE OPTIMIZATION
# AFRICAN COUNTRY DATA IMPUTER
# STREAMLIT APPLICATION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.ensemble import RandomForestRegressor


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="African Data Model-Based RL Imputer",
    page_icon="🌍",
    layout="wide"
)


# ==========================================
# 1. AFRICAN COUNTRIES — ISO-3
# ==========================================

ALL_AFRICA = {
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR",
    "CAF", "TCD", "COM", "COG", "COD", "CIV", "DJI", "EGY",
    "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN",
    "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI",
    "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
    "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN",
    "TZA", "TGO", "TUN", "UGA", "ZMB", "ZWE"
}


# ==========================================
# AFRICA ANALYSIS YEARS
# ==========================================

AFRICA_YEARS = [
    str(year)
    for year in range(2015, 2026)
]


# ==========================================
# VERIFY CONFIGURATION
# ==========================================

if len(ALL_AFRICA) != 54:

    raise RuntimeError(
        f"Expected 54 African countries, "
        f"found {len(ALL_AFRICA)}."
    )


# ==========================================
# 2. DETECT MISSING AFRICAN COUNTRIES
# ==========================================

def detect_missing_african_countries(
    df,
    country_col="geoUnit"
):
    """
    Compare countries in the uploaded dataset
    against the complete African ISO-3 country list.
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


# ==========================================
# 3. CREATE MISSING AFRICAN COUNTRIES TABLE
# ==========================================

def create_missing_africa_table(
    missing_countries,
    year_cols=None
):
    """
    Create an empty table for African countries
    that are completely absent.
    """

    if year_cols is None:
        year_cols = AFRICA_YEARS

    missing_table = pd.DataFrame({
        "geoUnit": missing_countries
    })

    for year in year_cols:

        missing_table[year] = np.nan

    return missing_table


# ==========================================
# 4. TRAIN COUNTRY-LEVEL WORLD MODEL
# ==========================================

def train_country_world_model(
    df,
    country_col="geoUnit",
    year_cols=None,
    n_estimators=500
):
    """
    Train a cross-country temporal World Model.

    Learns:

        [t-3, t-2, t-1] -> t
    """

    if year_cols is None:
        year_cols = AFRICA_YEARS

    data = df.copy()

    # --------------------------------------
    # Convert years to numeric
    # --------------------------------------

    for col in year_cols:

        if col not in data.columns:

            raise ValueError(
                f"Required year column "
                f"'{col}' was not found."
            )

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # --------------------------------------
    # Create temporal samples
    # --------------------------------------

    for _, row in data.iterrows():

        values = row[year_cols].to_numpy(
            dtype=float
        )

        for i in range(3, len(values)):

            state = values[i - 3:i]

            target = values[i]

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

    # --------------------------------------
    # Random Forest World Model
    # --------------------------------------

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


# ==========================================
# 5. INITIALIZE COMPLETELY MISSING COUNTRIES
# ==========================================

def initialize_missing_african_countries(
    df,
    missing_table,
    world_model,
    year_cols=None
):
    """
    Initialize trajectories for African countries
    that are completely absent from the uploaded data.

    The initialization uses cross-country information
    from the uploaded dataset.

    The World Model is then used to generate the
    missing country's temporal trajectory.

    Existing observations in the uploaded dataset
    are NOT modified.
    """

    if year_cols is None:
        year_cols = AFRICA_YEARS

    # --------------------------------------
    # Copy the missing-country table
    # --------------------------------------

    result = missing_table.copy()

    if len(result) == 0:
        return result

    # --------------------------------------
    # Convert source data to numeric
    # --------------------------------------

    source = df.copy()

    for year in year_cols:

        source[year] = pd.to_numeric(
            source[year],
            errors="coerce"
        )

    # --------------------------------------
    # Calculate yearly cross-country medians
    # --------------------------------------

    yearly_medians = {}

    for year in year_cols:

        values = source[year].dropna()

        if len(values) > 0:

            yearly_medians[year] = float(
                values.median()
            )

        else:

            yearly_medians[year] = np.nan

    # --------------------------------------
    # Calculate overall median
    # --------------------------------------

    all_values = (
        source[year_cols]
        .to_numpy(dtype=float)
        .flatten()
    )

    all_values = all_values[
        ~np.isnan(all_values)
    ]

    if len(all_values) == 0:

        raise ValueError(
            "The uploaded dataset contains no "
            "numeric observations from which to "
            "initialize missing countries."
        )

    global_median = float(
        np.median(all_values)
    )

    # --------------------------------------
    # Initialize each missing country
    # --------------------------------------

    for idx in result.index:

        values = np.full(
            len(year_cols),
            np.nan,
            dtype=float
        )

        # ----------------------------------
        # Create initial 2015–2017 state
        # ----------------------------------

        initial_state = []

        for year in year_cols[:3]:

            value = yearly_medians.get(
                year,
                np.nan
            )

            if np.isnan(value):

                value = global_median

            initial_state.append(value)

        values[:3] = np.asarray(
            initial_state,
            dtype=float
        )

        # ----------------------------------
        # Generate future years
        # ----------------------------------

        for j in range(3, len(year_cols)):

            state = values[j - 3:j]

            if np.isnan(state).any():

                prediction = global_median

            else:

                prediction = world_model.predict(
                    state.reshape(1, -1)
                )[0]

            # Safety check
            if not np.isfinite(prediction):

                prediction = global_median

            values[j] = prediction

        # ----------------------------------
        # Save trajectory
        # ----------------------------------

        result.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

    return result


# ==========================================
# MAIN STREAMLIT APPLICATION
# ==========================================

st.title(
    "🌍 Model-Based RL + Online Optimization"
)

st.header(
    "African Country Missing Data Imputer"
)

st.write(
    """
    Upload an Excel dataset and the application
    will detect missing African countries and
    generate 2015–2025 values using a
    cross-country World Model followed by
    Model-Based RL and Online Optimization.
    """
)


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📂 Upload your Excel file",
    type=["xlsx", "xls"]
)


# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    # ======================================
    # READ EXCEL
    # ======================================

    try:

        df = pd.read_excel(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read Excel file: {e}"
        )

        st.stop()


    # ======================================
    # STANDARDIZE COLUMN NAMES
    # ======================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # ======================================
    # CHECK geoUnit
    # ======================================

    if "geoUnit" not in df.columns:

        st.error(
            "The uploaded Excel file must "
            "contain a 'geoUnit' column."
        )

        st.stop()


    # ======================================
    # CHECK YEAR COLUMNS
    # ======================================

    missing_year_columns = [
        year
        for year in AFRICA_YEARS
        if year not in df.columns
    ]

    if missing_year_columns:

        st.error(
            "The following required year columns "
            f"are missing: {missing_year_columns}"
        )

        st.stop()


    # ======================================
    # CONVERT YEARS TO NUMERIC
    # ======================================

    for year in AFRICA_YEARS:

        df[year] = pd.to_numeric(
            df[year],
            errors="coerce"
        )


    # ======================================
    # DISPLAY UPLOADED DATA
    # ======================================

    st.subheader(
        "📊 Uploaded Dataset"
    )

    st.write(
        f"Rows: {df.shape[0]:,}"
    )

    st.write(
        f"Columns: {df.shape[1]:,}"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


    # ======================================
    # STEP 2
    # DETECT MISSING AFRICAN COUNTRIES
    # ======================================

    st.subheader(
        "🌍 Step 2 — African Country Coverage"
    )

    try:

        missing_countries = (
            detect_missing_african_countries(
                df,
                country_col="geoUnit"
            )
        )

    except ValueError as e:

        st.error(
            str(e)
        )

        st.stop()


    if len(missing_countries) > 0:

        st.warning(
            f"{len(missing_countries)} African "
            "countries are completely absent."
        )

        st.dataframe(
            pd.DataFrame({
                "Missing African ISO-3":
                    missing_countries
            }),
            use_container_width=True
        )

    else:

        st.success(
            "All 54 African countries are present."
        )


    # ======================================
    # STEP 3
    # CREATE MISSING COUNTRY TABLE
    # ======================================

    st.subheader(
        "📋 Step 3 — Missing Country Table"
    )

    missing_table = (
        create_missing_africa_table(
            missing_countries,
            AFRICA_YEARS
        )
    )

    if len(missing_table) > 0:

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    else:

        st.info(
            "No completely missing African "
            "countries detected."
        )


    # ======================================
    # STEP 4
    # TRAIN WORLD MODEL
    # ======================================

    st.subheader(
        "🧠 Step 4 — Train Country World Model"
    )

    with st.spinner(
        "Training cross-country World Model..."
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


    # ======================================
    # STEP 5
    # INITIALIZE MISSING COUNTRIES
    # ======================================

    st.subheader(
        "🚀 Step 5 — Initialize Missing "
        "African Countries"
    )

    if len(missing_table) > 0:

        with st.spinner(
            "Generating initial country trajectories..."
        ):

            try:

                initialized_missing = (
                    initialize_missing_african_countries(
                        df,
                        missing_table,
                        world_model,
                        AFRICA_YEARS
                    )
                )

            except ValueError as e:

                st.error(
                    f"Initialization failed: {e}"
                )

                st.stop()


        st.success(
            "Initial trajectories generated "
            "successfully."
        )

        st.write(
            """
            The completely missing African countries
            have now been initialized using
            cross-country information and the
            trained World Model.
            """
        )

        st.dataframe(
            initialized_missing,
            use_container_width=True
        )

    else:

        initialized_missing = pd.DataFrame(
            columns=[
                "geoUnit"
            ] + AFRICA_YEARS
        )

        st.info(
            "No completely missing countries "
            "require initialization."
        )


    # ======================================
    # SHOW INITIALIZED COUNTRY COUNT
    # ======================================

    st.metric(
        "Completely Missing Countries Initialized",
        len(initialized_missing)
    )
