# ============================================================
# MODEL-BASED RL + ONLINE OPTIMIZATION
# AFRICAN COUNTRY EXCEL IMPUTER
# Streamlit Community Cloud
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
    page_icon="🔄",
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

AFRICA_YEARS = [
    str(y) for y in range(2015, 2026)
]


# ============================================================
# GENERAL SETTINGS
# ============================================================

DEFAULT_ALPHA = 0.08
DEFAULT_EPISODES = 200
DEFAULT_TREES = 500


# ============================================================
# SAFE WORLD-MODEL PREDICTION
# ============================================================

def safe_prediction(model, state):
    """
    Always return a single scalar float from the Random Forest.

    This prevents errors such as:

        ValueError:
        setting an array element with a sequence
    """

    state = np.asarray(
        state,
        dtype=float
    ).reshape(1, -1)

    prediction = model.predict(state)

    prediction = np.asarray(
        prediction,
        dtype=float
    ).reshape(-1)

    if len(prediction) == 0:
        return np.nan

    return float(prediction[0])


# ============================================================
# DETECT YEAR COLUMNS
# ============================================================

def detect_year_columns(
    df,
    start_year=2015,
    end_year=2025
):

    columns = [
        str(c)
        for c in df.columns
    ]

    years = [
        str(y)
        for y in range(
            start_year,
            end_year + 1
        )
    ]

    available = [
        y for y in years
        if y in columns
    ]

    return available


# ============================================================
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
# TRAIN COUNTRY-LEVEL WORLD MODEL
# ============================================================

def train_country_world_model(
    df,
    country_col="geoUnit",
    year_cols=None,
    n_estimators=500
):

    if year_cols is None:
        year_cols = AFRICA_YEARS

    data = df.copy()

    # --------------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------------

    valid_year_cols = [
        col for col in year_cols
        if col in data.columns
    ]

    if len(valid_year_cols) < 4:
        raise ValueError(
            "At least four year columns are required "
            "to train the temporal world model."
        )

    # --------------------------------------------------------
    # Convert years to numeric
    # --------------------------------------------------------

    for col in valid_year_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # --------------------------------------------------------
    # Temporal training samples
    #
    # Example:
    #
    # 2015 2016 2017 -> 2018
    # 2016 2017 2018 -> 2019
    # 2017 2018 2019 -> 2020
    # ...
    # --------------------------------------------------------

    for _, row in data.iterrows():

        values = row[
            valid_year_cols
        ].to_numpy(
            dtype=float
        )

        for i in range(
            3,
            len(values)
        ):

            state = values[
                i - 3:i
            ]

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
# EMPIRICAL COUNTRY-SPECIFIC INITIAL STATE
# ============================================================

def estimate_country_initial_state(
    reference_df,
    year_cols,
    country_code
):
    """
    Estimate a distinct initial 2015-2017 state for a
    completely missing country.

    The model has no actual observations for a country that
    is completely absent. Therefore, its initial state is
    sampled reproducibly from the empirical distributions
    of African countries already present in the dataset.

    The country ISO-3 code creates a deterministic seed so
    different countries receive different starting states.
    """

    data = reference_df[
        year_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Reproducible country-specific random generator
    # --------------------------------------------------------

    seed_bytes = hashlib.sha256(
        country_code.encode(
            "utf-8"
        )
    ).digest()

    seed = int.from_bytes(
        seed_bytes[:8],
        byteorder="little"
    ) % (2**32 - 1)

    rng = np.random.default_rng(
        seed
    )

    global_values = data.to_numpy(
        dtype=float
    )

    global_mean = np.nanmean(
        global_values
    )

    if np.isnan(global_mean):
        global_mean = 0.0

    initial_values = []

    # --------------------------------------------------------
    # Estimate each of first three years independently
    # --------------------------------------------------------

    for year in year_cols[:3]:

        column = pd.to_numeric(
            data[year],
            errors="coerce"
        ).dropna()

        if len(column) == 0:
            initial_values.append(
                global_mean
            )
            continue

        # ----------------------------------------------------
        # Bootstrap empirical observation
        # ----------------------------------------------------

        selected = float(
            rng.choice(
                column.to_numpy()
            )
        )

        # Small controlled perturbation prevents identical
        # values while keeping the estimate near the empirical
        # distribution.
        std = float(
            column.std()
        )

        if np.isnan(std):
            std = 0.0

        perturbation = rng.normal(
            0,
            max(
                abs(selected) * 0.02,
                std * 0.02,
                1e-9
            )
        )

        value = selected + perturbation

        # Prevent negative values when the underlying dataset
        # represents non-negative quantities.
        if column.min() >= 0:
            value = max(
                0.0,
                value
            )

        initial_values.append(
            float(value)
        )

    return np.asarray(
        initial_values,
        dtype=float
    )


# ============================================================
# GENERATE COMPLETELY MISSING COUNTRY TRAJECTORY
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
    Generate a complete 2015-2025 trajectory for a country
    that is completely absent from the uploaded dataset.

    Model-Based RL:
        Previous three years -> predicted next year

    Online Optimization:
        x_new = x_old + alpha(prediction - x_old)
    """

    if len(year_cols) < 4:
        raise ValueError(
            "At least four years are required."
        )

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # Country-specific initial state
    # --------------------------------------------------------

    initial_values = estimate_country_initial_state(
        reference_df,
        year_cols,
        country_code
    )

    values[:3] = initial_values

    # --------------------------------------------------------
    # Model-Based RL trajectory
    # --------------------------------------------------------

    for j in range(
        3,
        len(values)
    ):

        state = values[
            j - 3:j
        ]

        if np.isnan(state).any():
            continue

        prediction = safe_prediction(
            world_model,
            state
        )

        if np.isnan(prediction):
            continue

        values[j] = prediction

    # --------------------------------------------------------
    # Online Optimization
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

            if np.isnan(state).any():
                continue

            prediction = safe_prediction(
                world_model,
                state
            )

            if np.isnan(prediction):
                continue

            values[j] = (
                values[j]
                + alpha
                * (
                    prediction
                    - values[j]
                )
            )

        difference = np.nanmax(
            np.abs(
                values
                - previous
            )
        )

        if (
            np.isnan(difference)
            or difference < 1e-6
        ):
            break

    return np.round(
        values,
        3
    )


# ============================================================
# IMPUTE MISSING VALUES FOR EXISTING COUNTRIES
# ============================================================

def impute_existing_country_missing_values(
    df,
    world_model,
    year_cols,
    alpha=0.08,
    episodes=200
):
    """
    Impute missing years for countries already present.

    IMPORTANT:
    Existing observed values are NEVER changed.
    Only originally missing values are updated.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Ensure numeric years
    # --------------------------------------------------------

    for col in year_cols:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Original missing-value mask
    # --------------------------------------------------------

    original_missing = (
        result[year_cols]
        .isna()
    )

    total_imputed = 0

    # --------------------------------------------------------
    # Process each country
    # --------------------------------------------------------

    for idx in result.index:

        values = result.loc[
            idx,
            year_cols
        ].to_numpy(
            dtype=float
        )

        mask = original_missing.loc[
            idx,
            year_cols
        ].to_numpy(
            dtype=bool
        )

        for _ in range(
            episodes
        ):

            previous = values.copy()

            for j in range(
                len(values)
            ):

                # NEVER modify observed values
                if not mask[j]:
                    continue

                prediction = np.nan

                # =================================================
                # 1. WORLD MODEL
                # =================================================

                if j >= 3:

                    state = values[
                        j - 3:j
                    ]

                    if not np.isnan(
                        state
                    ).any():

                        prediction = safe_prediction(
                            world_model,
                            state
                        )

                # =================================================
                # 2. NEAREST NEIGHBOURS
                # =================================================

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

                # =================================================
                # 3. ROW MEAN
                # =================================================

                if np.isnan(
                    prediction
                ):

                    valid_values = values[
                        ~np.isnan(values)
                    ]

                    if len(
                        valid_values
                    ) > 0:

                        prediction = float(
                            np.mean(
                                valid_values
                            )
                        )

                # =================================================
                # 4. GLOBAL MEAN
                # =================================================

                if np.isnan(
                    prediction
                ):

                    all_values = (
                        result[year_cols]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    prediction = float(
                        np.nanmean(
                            all_values
                        )
                    )

                # =================================================
                # CRITICAL FIX
                # =================================================

                if not np.isnan(
                    prediction
                ):

                    prediction = float(
                        np.asarray(
                            prediction
                        ).reshape(-1)[0]
                    )

                    # First assignment
                    if np.isnan(
                        values[j]
                    ):

                        values[j] = prediction

                    # Online optimization
                    else:

                        values[j] = (
                            values[j]
                            + alpha
                            * (
                                prediction
                                - values[j]
                            )
                        )

                    total_imputed += 1

            # ----------------------------------------------------
            # Convergence
            # ----------------------------------------------------

            if np.all(
                ~np.isnan(values)
            ):
                break

            valid_difference = (
                np.abs(
                    values
                    - previous
                )
            )

            valid_difference = (
                valid_difference[
                    ~np.isnan(
                        valid_difference
                    )
                ]
            )

            if len(
                valid_difference
            ) == 0:
                break

            if np.max(
                valid_difference
            ) < 1e-6:
                break

        # --------------------------------------------------------
        # Save country
        # --------------------------------------------------------

        result.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

    return result, total_imputed


# ============================================================
# BUILD COMPLETELY MISSING AFRICAN COUNTRIES
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
    Create and populate countries that are completely absent
    from the uploaded dataset.
    """

    if not missing_countries:
        return df.copy()

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

        for j, year in enumerate(
            year_cols
        ):

            row[year] = float(
                trajectory[j]
            )

        added_rows.append(
            row
        )

    added_df = pd.DataFrame(
        added_rows
    )

    # --------------------------------------------------------
    # Preserve original columns
    # --------------------------------------------------------

    output = df.copy()

    for col in year_cols:

        if col not in output.columns:
            output[col] = np.nan

    # Ensure country column is first
    columns = [
        "geoUnit"
    ] + [
        c for c in output.columns
        if c != "geoUnit"
    ]

    output = output[
        columns
    ]

    # Add missing countries
    output = pd.concat(
        [
            output,
            added_df
        ],
        ignore_index=True
    )

    return output


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

st.title(
    "🔄 African Model-Based RL Imputer"
)

st.markdown(
    """
    ### Model-Based Reinforcement Learning + Online Optimization

    Upload an Excel dataset containing African countries and
    annual observations. The application:

    1. Detects missing years for countries already present.
    2. Trains a country-level temporal World Model.
    3. Imputes missing years using Model-Based RL.
    4. Applies online optimization.
    5. Detects completely missing African countries.
    6. Generates 2015–2025 trajectories for those countries.
    7. Produces a downloadable Excel workbook.
    """
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "⚙️ Model Settings"
)

start_year = st.sidebar.number_input(
    "Start year",
    min_value=1900,
    max_value=2100,
    value=2015,
    step=1
)

end_year = st.sidebar.number_input(
    "End year",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

n_estimators = st.sidebar.slider(
    "Random Forest trees",
    min_value=100,
    max_value=1000,
    value=500,
    step=100
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


# ============================================================
# UPLOAD EXCEL
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload Excel dataset",
    type=[
        "xlsx",
        "xls"
    ]
)


# ============================================================
# APPLICATION AFTER UPLOAD
# ============================================================

if uploaded_file is not None:

    try:

        # ------------------------------------------------------
        # READ EXCEL
        # ------------------------------------------------------

        if uploaded_file.name.lower().endswith(
            ".xls"
        ):

            df = pd.read_excel(
                uploaded_file,
                engine="xlrd"
            )

        else:

            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        # ------------------------------------------------------
        # NORMALIZE COLUMN NAMES
        # ------------------------------------------------------

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        # ------------------------------------------------------
        # CHECK COUNTRY COLUMN
        # ------------------------------------------------------

        if "geoUnit" not in df.columns:

            st.error(
                "The uploaded Excel file must contain "
                "a 'geoUnit' column containing ISO-3 "
                "country codes."
            )

            st.stop()

        # ------------------------------------------------------
        # DETECT YEARS
        # ------------------------------------------------------

        year_cols = detect_year_columns(
            df,
            start_year=int(start_year),
            end_year=int(end_year)
        )

        if len(year_cols) < 4:

            st.error(
                "At least four annual columns are required "
                "for Model-Based RL."
            )

            st.stop()

        # ------------------------------------------------------
        # Convert year values
        # ------------------------------------------------------

        for col in year_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ------------------------------------------------------
        # DATASET INFORMATION
        # ------------------------------------------------------

        st.subheader(
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
                "Years detected",
                len(year_cols)
            )

        st.write(
            "Detected years:"
        )

        st.write(
            ", ".join(year_cols)
        )

        # ------------------------------------------------------
        # PREVIEW
        # ------------------------------------------------------

        with st.expander(
            "View uploaded dataset"
        ):

            st.dataframe(
                df,
                use_container_width=True
            )

        # ======================================================
        # STEP 1 — DETECT MISSING AFRICAN COUNTRIES
        # ======================================================

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

            st.dataframe(
                pd.DataFrame({
                    "ISO-3":
                    missing_countries
                }),
                use_container_width=True
            )

        else:

            st.success(
                "All African ISO-3 countries are present."
            )

        # ======================================================
        # STEP 2 — TRAIN WORLD MODEL
        # ======================================================

        st.subheader(
            "🧠 Country-Level World Model"
        )

        progress = st.progress(
            0
        )

        status = st.empty()

        status.info(
            "Training temporal World Model..."
        )

        world_model, training_samples = (
            train_country_world_model(
                df=df,
                country_col="geoUnit",
                year_cols=year_cols,
                n_estimators=int(
                    n_estimators
                )
            )
        )

        progress.progress(
            25
        )

        st.success(
            "Country-level World Model "
            "trained successfully."
        )

        st.info(
            f"Temporal training samples: "
            f"{training_samples:,}"
        )

        # ======================================================
        # STEP 3 — IMPUTE EXISTING COUNTRIES
        # ======================================================

        st.subheader(
            "🔄 Imputing Missing Years"
        )

        status.info(
            "Step 1/2: Imputing missing years "
            "for countries already present..."
        )

        df_imputed, existing_imputed_count = (
            impute_existing_country_missing_values(
                df=df,
                world_model=world_model,
                year_cols=year_cols,
                alpha=float(alpha),
                episodes=int(episodes)
            )
        )

        progress.progress(
            60
        )

        st.success(
            "Existing-country missing values "
            "completed."
        )

        st.metric(
            "Existing-country values imputed",
            existing_imputed_count
        )

        # ======================================================
        # STEP 4 — ADD COMPLETELY MISSING AFRICAN COUNTRIES
        # ======================================================

        st.subheader(
            "🌍 Adding Completely Missing African Countries"
        )

        status.info(
            "Step 2/2: Generating trajectories "
            "for missing African countries..."
        )

        df_final = add_missing_african_countries(
            df=df_imputed,
            missing_countries=missing_countries,
            world_model=world_model,
            year_cols=year_cols,
            alpha=float(alpha),
            episodes=int(episodes)
        )

        progress.progress(
            100
        )

        status.success(
            "Processing completed."
        )

        # ======================================================
        # STEP 5 — RESULTS
        # ======================================================

        st.subheader(
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
                "World-model samples",
                training_samples
            )

        with col4:

            remaining_missing = int(
                df_final[
                    year_cols
                ].isna().sum().sum()
            )

            st.metric(
                "Remaining missing values",
                remaining_missing
            )

        # ======================================================
        # SHOW ADDED COUNTRIES
        # ======================================================

        if missing_countries:

            st.subheader(
                "➕ Added African Countries"
            )

            added_display = df_final[
                df_final[
                    "geoUnit"
                ].isin(
                    missing_countries
                )
            ][
                [
                    "geoUnit"
                ] + year_cols
            ]

            st.dataframe(
                added_display,
                use_container_width=True
            )

        # ======================================================
        # SHOW COMPLETE RESULT
        # ======================================================

        st.subheader(
            "📋 Final Imputed Dataset"
        )

        st.dataframe(
            df_final,
            use_container_width=True
        )

        # ======================================================
        # MISSING VALUE CHECK
        # ======================================================

        st.subheader(
            "🔍 Missing-Value Validation"
        )

        missing_before = int(
            df[
                year_cols
            ].isna().sum().sum()
        )

        missing_after = int(
            df_final[
                year_cols
            ].isna().sum().sum()
        )

        validation_df = pd.DataFrame({
            "Metric": [
                "Missing values before",
                "Missing values after",
                "Values imputed"
            ],
            "Value": [
                missing_before,
                missing_after,
                missing_before
                - missing_after
            ]
        })

        st.dataframe(
            validation_df,
            use_container_width=True,
            hide_index=True
        )

        if missing_after == 0:

            st.success(
                "✅ All 2015–2025 missing values "
                "have been populated."
            )

        else:

            st.warning(
                f"{missing_after} missing values "
                "remain."
            )

        # ======================================================
        # CREATE EXCEL OUTPUT
        # ======================================================

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            # --------------------------------------------------
            # Sheet 1
            # --------------------------------------------------

            df_final.to_excel(
                writer,
                index=False,
                sheet_name="Imputed_Africa"
            )

            # --------------------------------------------------
            # Sheet 2
            # --------------------------------------------------

            pd.DataFrame({
                "Missing African Countries":
                missing_countries
            }).to_excel(
                writer,
                index=False,
                sheet_name="Added_Countries"
            )

            # --------------------------------------------------
            # Sheet 3
            # --------------------------------------------------

            validation_df.to_excel(
                writer,
                index=False,
                sheet_name="Validation"
            )

            # --------------------------------------------------
            # Sheet 4 — Model Settings
            # --------------------------------------------------

            settings_df = pd.DataFrame({
                "Parameter": [
                    "Start Year",
                    "End Year",
                    "Random Forest Trees",
                    "Online Optimization Alpha",
                    "Optimization Episodes",
                    "World Model Training Samples",
                    "Existing Country Values Imputed",
                    "African Countries Added"
                ],
                "Value": [
                    start_year,
                    end_year,
                    n_estimators,
                    alpha,
                    episodes,
                    training_samples,
                    existing_imputed_count,
                    len(missing_countries)
                ]
            })

            settings_df.to_excel(
                writer,
                index=False,
                sheet_name="Model_Settings"
            )

        output.seek(
            0
        )

        # ======================================================
        # DOWNLOAD
        # ======================================================

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

    except Exception as e:

        st.error(
            "The application encountered an error "
            "while processing the uploaded dataset."
        )

        st.exception(e)
