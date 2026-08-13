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
# CHECK AFRICA CONFIGURATION
# ==========================================

if len(ALL_AFRICA) != 54:
    raise RuntimeError(
        f"Expected 54 African countries, "
        f"but found {len(ALL_AFRICA)}."
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

    Returns a sorted list of African countries
    that are completely absent from the dataset.
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
    that are completely absent from the dataset.

    Values from 2015–2025 are initialized as NaN.
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

    The model learns:

        [t-3, t-2, t-1] -> t

    using temporal observations from all countries
    available in the uploaded dataset.

    This allows the model to generate a trajectory
    for a completely missing African country.
    """

    if year_cols is None:
        year_cols = AFRICA_YEARS

    data = df.copy()

    # --------------------------------------
    # Validate year columns
    # --------------------------------------

    for col in year_cols:

        if col not in data.columns:

            raise ValueError(
                f"Required year column '{col}' "
                "was not found."
            )

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # --------------------------------------
    # Create temporal training samples
    # --------------------------------------

    for _, row in data.iterrows():

        values = row[year_cols].to_numpy(
            dtype=float
        )

        for i in range(3, len(values)):

            state = values[i - 3:i]

            target = values[i]

            # Require three complete previous years
            if np.isnan(state).any():
                continue

            # Target must be available
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

    # --------------------------------------
    # Validate training samples
    # --------------------------------------

    if len(X) < 5:

        raise ValueError(
            "Not enough complete temporal observations "
            "to train the world model."
        )

    # --------------------------------------
    # Train Random Forest World Model
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
    Upload an Excel dataset and the application will:
    
    1. Detect African countries completely missing
       from the dataset.
    2. Detect missing yearly observations.
    3. Train a cross-country temporal World Model.
    4. Generate missing country trajectories.
    5. Apply Model-Based Reinforcement Learning.
    6. Apply Online Optimization.
    7. Produce a completed African dataset for
       2015–2025.
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
# PROCESS UPLOADED FILE
# ==========================================

if uploaded_file is not None:

    # ======================================
    # READ EXCEL FILE
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
    # SHOW UPLOADED DATASET
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
    # CHECK COUNTRY COLUMN
    # ======================================

    if "geoUnit" not in df.columns:

        st.error(
            """
            The uploaded Excel file must contain
            a column named `geoUnit`.

            Example:

            geoUnit | 2015 | 2016 | ... | 2025
            """
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
    # CONVERT YEAR VALUES TO NUMERIC
    # ======================================

    for year in AFRICA_YEARS:

        df[year] = pd.to_numeric(
            df[year],
            errors="coerce"
        )


    # ======================================
    # STEP 2
    # DETECT COMPLETELY MISSING COUNTRIES
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
            "countries are completely absent "
            "from the uploaded dataset."
        )

        missing_display = pd.DataFrame({
            "Missing African ISO-3":
                missing_countries
        })

        st.dataframe(
            missing_display,
            use_container_width=True
        )

    else:

        st.success(
            "All 54 African countries are present "
            "in the uploaded dataset."
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

        st.write(
            """
            These countries have no observations
            in the uploaded dataset. Their
            2015–2025 values will be generated
            by the model.
            """
        )

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    else:

        st.info(
            "No completely missing African "
            "countries were detected."
        )


    # ======================================
    # STEP 4
    # TRAIN COUNTRY WORLD MODEL
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
    # CURRENT MISSING VALUES
    # ======================================

    st.subheader(
        "📉 Current Missing Values"
    )

    missing_by_year = (
        df[AFRICA_YEARS]
        .isna()
        .sum()
    )

    st.dataframe(
        missing_by_year
        .to_frame("Missing Values"),
        use_container_width=True
    )


    total_missing = (
        df[AFRICA_YEARS]
        .isna()
        .sum()
        .sum()
    )

    st.metric(
        "Total Missing Values",
        f"{total_missing:,}"
    )


    # ======================================
    # MODEL STATUS
    # ======================================

    st.success(
        """
        Steps 1–4 completed successfully.

        The cross-country World Model is now ready
        to generate trajectories for completely
        missing African countries.
        """
    )
