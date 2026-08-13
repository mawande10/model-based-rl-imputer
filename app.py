# ============================================================
# AFRICAN MODEL-BASED RL + ONLINE OPTIMIZATION IMPUTER
# Streamlit Community Cloud Application
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
# AFRICAN COUNTRY ISO-3 LIST
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
    This application performs missing-data completion for African
    countries using a country-level temporal world model based on
    Random Forest regression combined with Model-Based Reinforcement
    Learning and Online Optimization.

    **Target period:** 2015–2025
    """
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Settings")

start_year = st.sidebar.number_input(
    "Start year",
    min_value=2015,
    max_value=2025,
    value=2015,
    step=1
)

end_year = st.sidebar.number_input(
    "End year",
    min_value=2015,
    max_value=2025,
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
# VALIDATE YEAR RANGE
# ============================================================

if start_year >= end_year:

    st.sidebar.error(
        "End year must be greater than start year."
    )

    st.stop()


SELECTED_YEARS = [
    str(y)
    for y in range(
        int(start_year),
        int(end_year) + 1
    )
]


# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Excel dataset",
    type=["xlsx", "xls"]
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

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    detected = [
        c for c in df.columns
        if c.isdigit()
    ]

    detected = sorted(
        detected,
        key=int
    )

    return detected


# ============================================================
# FUNCTION 3
# CLEAN NUMERIC YEAR DATA
# ============================================================

def prepare_numeric_year_data(
    df,
    year_cols
):

    result = df.copy()

    for col in year_cols:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    return result


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
    Train a temporal world model using three consecutive
    years to predict the following year.

    Example:

        2015, 2016, 2017 -> 2018
        2016, 2017, 2018 -> 2019
        2017, 2018, 2019 -> 2020

    The model is trained across all available countries.
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    data = df.copy()

    for col in year_cols:

        if col not in data.columns:

            data[col] = np.nan

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    X = []
    y = []

    # --------------------------------------------------------
    # CREATE TEMPORAL TRAINING SAMPLES
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
            "to train the world model. At least 5 "
            "training samples are required."
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

    return model, len(X)


# ============================================================
# FUNCTION 5
# SAFE WORLD MODEL PREDICTION
# ============================================================

def safe_world_model_prediction(
    world_model,
    state
):
    """
    Always return a scalar float.

    This prevents errors such as:

        values[j] = prediction

    when prediction is returned as an array.
    """

    state = np.asarray(
        state,
        dtype=float
    ).reshape(1, -1)

    prediction = world_model.predict(
        state
    )

    prediction = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return prediction


# ============================================================
# FUNCTION 6
# INITIAL STATE FROM AFRICAN DATA
# ============================================================

def create_country_specific_initial_state(
    reference_df,
    year_cols,
    country_code
):
    """
    Create a different initial 3-year state for each
    completely missing African country.

    Instead of assigning the same global median to every
    country, the method samples from the empirical African
    distributions for the first three years.

    A deterministic seed based on the ISO-3 country code
    ensures reproducibility.
    """

    data = reference_df[
        year_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # COUNTRY-SPECIFIC DETERMINISTIC RANDOM SEED
    # --------------------------------------------------------

    hash_value = hashlib.md5(
        country_code.encode(
            "utf-8"
        )
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
    # GLOBAL FALLBACK
    # --------------------------------------------------------

    global_values = data.to_numpy(
        dtype=float
    )

    global_mean = np.nanmean(
        global_values
    )

    if np.isnan(global_mean):

        global_mean = 0.0

    # --------------------------------------------------------
    # GENERATE DIFFERENT INITIAL VALUES
    # --------------------------------------------------------

    for year in year_cols[:3]:

        column = data[
            year
        ].dropna().to_numpy(
            dtype=float
        )

        if len(column) == 0:

            initial_values.append(
                global_mean
            )

        else:

            # Random empirical sample
            sampled_value = rng.choice(
                column
            )

            # Small country-specific perturbation
            std = np.nanstd(
                column
            )

            if np.isnan(std) or std == 0:

                std = abs(
                    sampled_value
                ) * 0.02

                if std == 0:
                    std = 0.01

            variation = rng.normal(
                0,
                std * 0.05
            )

            initial_values.append(
                float(
                    sampled_value
                    + variation
                )
            )

    # --------------------------------------------------------
    # FINAL NaN PROTECTION
    # --------------------------------------------------------

    initial_values = [

        global_mean
        if np.isnan(x)
        else float(x)

        for x in initial_values

    ]

    return np.asarray(
        initial_values,
        dtype=float
    )


# ============================================================
# FUNCTION 7
# GENERATE MISSING COUNTRY TRAJECTORY
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
    Generate a complete trajectory for a completely missing
    African country.

    Model-Based RL:
        Uses the trained temporal world model to predict
        future years.

    Online Optimization:
        Iteratively moves predicted values toward the
        world-model optimum using alpha.
    """

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_values = (
        create_country_specific_initial_state(
            reference_df,
            year_cols,
            country_code
        )
    )

    values[:3] = initial_values

    # --------------------------------------------------------
    # MODEL-BASED RL FORWARD TRAJECTORY
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

            continue

        prediction = (
            safe_world_model_prediction(
                world_model,
                state
            )
        )

        values[j] = prediction

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

            prediction = (
                safe_world_model_prediction(
                    world_model,
                    state
                )
            )

            # ------------------------------------------------
            # ONLINE UPDATE
            # ------------------------------------------------

            values[j] = (
                values[j]
                + alpha
                * (
                    prediction
                    - values[j]
                )
            )

        valid_difference = (
            np.abs(
                values
                - previous
            )
        )

        valid_difference = (
            valid_difference[
                np.isfinite(
                    valid_difference
                )
            ]
        )

        if len(
            valid_difference
        ) == 0:

            break

        difference = np.max(
            valid_difference
        )

        if difference < 1e-6:

            break

    return np.round(
        values,
        3
    )


# ============================================================
# FUNCTION 8
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
    Impute missing years for countries that already exist
    in the uploaded dataset.

    IMPORTANT:
    Observed values are never changed.

    Only cells that were originally NaN are modified.
    """

    result = df.copy()

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in year_cols:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMEMBER ORIGINAL MISSING VALUES
    # --------------------------------------------------------

    missing_mask = (
        result[
            year_cols
        ].isna()
    )

    # --------------------------------------------------------
    # PROCESS EACH COUNTRY
    # --------------------------------------------------------

    for idx in result.index:

        values = (
            result.loc[
                idx,
                year_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        original_mask = (
            missing_mask.loc[
                idx,
                year_cols
            ]
            .to_numpy(
                dtype=bool
            )
        )

        # ----------------------------------------------------
        # ITERATIVE MODEL-BASED RL
        # ----------------------------------------------------

        for _ in range(
            episodes
        ):

            previous = values.copy()

            changed = False

            # ------------------------------------------------
            # PROCESS MISSING YEARS
            # ------------------------------------------------

            for j in range(
                len(values)
            ):

                # NEVER MODIFY OBSERVED VALUES
                if not original_mask[j]:

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

                        prediction = (
                            safe_world_model_prediction(
                                world_model,
                                state
                            )
                        )

                # =================================================
                # 2. NEAREST LEFT/RIGHT VALUES
                # =================================================

                if (
                    not np.isfinite(
                        prediction
                    )
                ):

                    left = None
                    right = None

                    for k in range(
                        j - 1,
                        -1,
                        -1
                    ):

                        if np.isfinite(
                            values[k]
                        ):

                            left = (
                                values[k]
                            )

                            break

                    for k in range(
                        j + 1,
                        len(values)
                    ):

                        if np.isfinite(
                            values[k]
                        ):

                            right = (
                                values[k]
                            )

                            break

                    # --------------------------------------------
                    # BOTH SIDES AVAILABLE
                    # --------------------------------------------

                    if (
                        left is not None
                        and
                        right is not None
                    ):

                        prediction = (
                            left
                            + right
                        ) / 2.0

                    # --------------------------------------------
                    # LEFT ONLY
                    # --------------------------------------------

                    elif (
                        left is not None
                    ):

                        prediction = left

                    # --------------------------------------------
                    # RIGHT ONLY
                    # --------------------------------------------

                    elif (
                        right is not None
                    ):

                        prediction = right

                # =================================================
                # 3. COUNTRY ROW MEAN
                # =================================================

                if (
                    not np.isfinite(
                        prediction
                    )
                ):

                    valid_values = values[
                        np.isfinite(
                            values
                        )
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

                if (
                    not np.isfinite(
                        prediction
                    )
                ):

                    all_values = (
                        result[
                            year_cols
                        ]
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
                # 5. FINAL SAFETY FALLBACK
                # =================================================

                if (
                    not np.isfinite(
                        prediction
                    )
                ):

                    prediction = 0.0

                # =================================================
                # MODEL-BASED RL + ONLINE OPTIMIZATION
                # =================================================

                if np.isnan(
                    values[j]
                ):

                    # Initial RL prediction
                    values[j] = (
                        prediction
                    )

                else:

                    # Online optimization
                    values[j] = (
                        values[j]
                        + alpha
                        * (
                            prediction
                            - values[j]
                        )
                    )

                # ------------------------------------------------
                # GUARANTEE SCALAR
                # ------------------------------------------------

                values[j] = float(
                    np.asarray(
                        values[j]
                    ).reshape(-1)[0]
                )

                changed = True

            # ----------------------------------------------------
            # STOP WHEN COMPLETE
            # ----------------------------------------------------

            if not np.isnan(
                values
            ).any():

                break

            # ----------------------------------------------------
            # CONVERGENCE TEST
            # ----------------------------------------------------

            difference = np.abs(
                values
                - previous
            )

            difference = (
                difference[
                    np.isfinite(
                        difference
                    )
                ]
            )

            if len(
                difference
            ) == 0:

                break

            if np.max(
                difference
            ) < 1e-6:

                break

            if not changed:

                break

        # --------------------------------------------------------
        # FINAL NAN PROTECTION
        # --------------------------------------------------------

        for j in range(
            len(values)
        ):

            if (
                original_mask[j]
                and
                not np.isfinite(
                    values[j]
                )
            ):

                valid = values[
                    np.isfinite(
                        values
                    )
                ]

                if len(valid) > 0:

                    values[j] = float(
                        np.mean(valid)
                    )

                else:

                    values[j] = 0.0

        # --------------------------------------------------------
        # SAVE COUNTRY
        # --------------------------------------------------------

        result.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

    return result


# ============================================================
# FUNCTION 9
# ADD COMPLETELY MISSING AFRICAN COUNTRIES
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
    Generate and append completely missing African countries.

    Each country receives a country-specific trajectory.
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

        for j, year in enumerate(
            year_cols
        ):

            row[year] = float(
                trajectory[j]
            )

        added_rows.append(
            row
        )

    if added_rows:

        added_df = pd.DataFrame(
            added_rows
        )

        # Ensure same columns
        for col in result.columns:

            if col not in added_df.columns:

                added_df[col] = np.nan

        # Preserve original column order
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
# FUNCTION 10
# CREATE MISSING COUNTRY TABLE
# ============================================================

def create_missing_country_table(
    missing_countries,
    year_cols
):

    missing_table = pd.DataFrame(
        {
            "geoUnit":
                missing_countries
        }
    )

    for year in year_cols:

        missing_table[year] = np.nan

    return missing_table


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload an Excel file from the sidebar to begin."
    )

    st.markdown(
        """
        ### What this application does

        **Step 1:** Reads the uploaded Excel dataset.

        **Step 2:** Detects the 2015–2025 year columns.

        **Step 3:** Trains a country-level temporal world model.

        **Step 4:** Imputes missing years for countries already
        present in the dataset.

        **Step 5:** Detects African countries completely absent
        from the dataset.

        **Step 6:** Generates different 2015–2025 trajectories
        for those missing countries.

        **Step 7:** Produces a downloadable Excel workbook.
        """
    )

    st.stop()


# ============================================================
# READ EXCEL FILE
# ============================================================

try:

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

except Exception as e:

    st.error(
        f"Could not read the Excel file: {e}"
    )

    st.stop()


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]


# ============================================================
# CHECK COUNTRY COLUMN
# ============================================================

if "geoUnit" not in df.columns:

    st.error(
        "The uploaded Excel file must contain "
        "a 'geoUnit' column containing ISO-3 country codes."
    )

    st.write(
        "Detected columns:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# DETECT YEAR COLUMNS
# ============================================================

detected_years = detect_year_columns(
    df
)


# ============================================================
# CHECK REQUIRED YEARS
# ============================================================

missing_year_columns = [
    year
    for year in SELECTED_YEARS
    if year not in detected_years
]


if missing_year_columns:

    st.error(
        "The following required year columns are missing "
        f"from the uploaded dataset: "
        f"{missing_year_columns}"
    )

    st.stop()


year_cols = SELECTED_YEARS.copy()


# ============================================================
# PREPARE DATA
# ============================================================

df = prepare_numeric_year_data(
    df,
    year_cols
)


# ============================================================
# DATASET INFORMATION
# ============================================================

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
        "Year columns",
        len(year_cols)
    )


with st.expander(
    "Preview uploaded dataset"
):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# AFRICAN COUNTRY COVERAGE
# ============================================================

st.subheader(
    "🌍 African Country Coverage"
)


missing_countries = (
    detect_missing_african_countries(
        df,
        "geoUnit"
    )
)


existing_africa = (
    set(
        df["geoUnit"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    & ALL_AFRICA
)


st.write(
    f"**African countries present:** "
    f"{len(existing_africa)} / {len(ALL_AFRICA)}"
)


if missing_countries:

    st.warning(
        f"{len(missing_countries)} African countries "
        "are completely absent from the uploaded dataset."
    )

    st.dataframe(
        pd.DataFrame(
            {
                "ISO-3":
                    missing_countries
            }
        ),
        use_container_width=True
    )

else:

    st.success(
        "All African ISO-3 countries are present."
    )


# ============================================================
# TRAIN WORLD MODEL
# ============================================================

st.subheader(
    "🧠 Country-Level World Model"
)


with st.spinner(
    "Training country-level temporal world model..."
):

    try:

        world_model, training_samples = (
            train_country_world_model(
                df=df,
                country_col="geoUnit",
                year_cols=year_cols,
                n_estimators=n_estimators
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


# ============================================================
# START PROCESSING
# ============================================================

st.subheader(
    "🔄 Imputing Missing Years"
)


progress = st.progress(
    0
)

status = st.empty()


# ============================================================
# STEP 1
# EXISTING COUNTRIES
# ============================================================

status.info(
    "Step 1/2: Imputing missing years for "
    "countries already present..."
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


# ============================================================
# CHECK REMAINING MISSING VALUES
# ============================================================

remaining_missing = (
    int(
        df_existing_imputed[
            year_cols
        ]
        .isna()
        .sum()
        .sum()
    )
)


st.info(
    f"Missing year cells remaining after "
    f"existing-country imputation: "
    f"{remaining_missing:,}"
)


# ============================================================
# STEP 2
# ADD COMPLETELY MISSING AFRICAN COUNTRIES
# ============================================================

status.info(
    "Step 2/2: Generating trajectories for "
    "completely missing African countries..."
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
    "Processing completed."
)


# ============================================================
# FINAL ROUNDING
# ============================================================

for col in year_cols:

    df_final[col] = pd.to_numeric(
        df_final[col],
        errors="coerce"
    ).round(3)


# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

final_missing = int(
    df_final[
        year_cols
    ]
    .isna()
    .sum()
    .sum()
)


# ============================================================
# RESULTS
# ============================================================

st.success(
    "Model-Based RL + Online Optimization completed."
)


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
        "World-model training samples",
        training_samples
    )


with col4:

    st.metric(
        "Remaining missing cells",
        final_missing
    )


# ============================================================
# SHOW ADDED COUNTRIES
# ============================================================

if missing_countries:

    st.subheader(
        "➕ Added African Countries"
    )

    added_country_df = (
        df_final[
            df_final[
                "geoUnit"
            ]
            .astype(str)
            .str.upper()
            .isin(
                missing_countries
            )
        ]
        .copy()
    )

    st.dataframe(
        added_country_df[
            [
                "geoUnit"
            ]
            + year_cols
        ],
        use_container_width=True
    )


# ============================================================
# SHOW EXISTING COUNTRY IMPUTATION
# ============================================================

st.subheader(
    "📋 Final Imputed Dataset"
)


st.dataframe(
    df_final,
    use_container_width=True,
    height=500
)


# ============================================================
# MISSING VALUE SUMMARY
# ============================================================

st.subheader(
    "🔍 Final Missing-Value Check"
)


missing_summary = pd.DataFrame(
    {
        "Year":
            year_cols,
        "Missing Values":
            [
                int(
                    df_final[
                        year
                    ].isna().sum()
                )
                for year in year_cols
            ]
    }
)


st.dataframe(
    missing_summary,
    use_container_width=True
)


if final_missing == 0:

    st.success(
        "✅ All requested African year values "
        "for the selected period have been populated."
    )

else:

    st.warning(
        f"{final_missing:,} missing cells remain. "
        "These could not be estimated from the available "
        "training information."
    )


# ============================================================
# VERIFY ADDED COUNTRY TRAJECTORIES ARE NOT IDENTICAL
# ============================================================

if len(missing_countries) > 1:

    added_values = (
        df_final[
            df_final[
                "geoUnit"
            ]
            .astype(str)
            .str.upper()
            .isin(
                missing_countries
            )
        ][
            year_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    unique_trajectories = len(
        {
            tuple(
                np.round(
                    row,
                    3
                )
            )
            for row in added_values
        }
    )

    st.info(
        f"Added-country trajectories: "
        f"{unique_trajectories} unique out of "
        f"{len(missing_countries)} countries."
    )


# ============================================================
# CREATE EXCEL OUTPUT
# ============================================================

output = io.BytesIO()


with pd.ExcelWriter(
    output,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # MAIN DATASET
    # --------------------------------------------------------

    df_final.to_excel(
        writer,
        index=False,
        sheet_name="Imputed_Africa"
    )

    # --------------------------------------------------------
    # ADDED COUNTRIES
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "Missing African Countries":
                missing_countries
        }
    ).to_excel(
        writer,
        index=False,
        sheet_name="Added_Countries"
    )

    # --------------------------------------------------------
    # FINAL MISSING SUMMARY
    # --------------------------------------------------------

    missing_summary.to_excel(
        writer,
        index=False,
        sheet_name="Missing_Value_Summary"
    )


output.seek(0)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "⬇️ Download Results"
)


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


# ============================================================
# METHODOLOGY INFORMATION
# ============================================================

with st.expander(
    "🧮 Model Methodology"
):

    st.markdown(
        """
        ### Model-Based RL + Online Optimization

        The temporal world model uses three consecutive years
        as the state:

        **State**

        `sₜ = [xₜ₋₃, xₜ₋₂, xₜ₋₁]`

        **World-model prediction**

        `x̂ₜ = f(sₜ)`

        where `f` is a Random Forest regression model trained
        using temporal observations from the uploaded dataset.

        **Online optimization**

        `xₜ ← xₜ + α(x̂ₜ − xₜ)`

        where:

        - `α` = online optimization learning rate
        - `xₜ` = current estimated value
        - `x̂ₜ` = world-model prediction

        ### Existing countries

        Only originally missing cells are modified.

        Observed values are preserved.

        ### Completely missing countries

        For an African country with no observations at all,
        the first three years are initialized using the empirical
        African data distribution.

        A deterministic country-specific seed is used so that
        different missing countries receive different initial
        states.

        The world model then generates the remaining trajectory
        from 2018 onward.

        ### Reproducibility

        The Random Forest uses:

        `random_state = 42`

        and the missing-country initialization uses a deterministic
        ISO-3-specific seed.
        """
    )
