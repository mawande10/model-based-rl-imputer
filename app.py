import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model-Based RL + Online Optimization Imputer",
    page_icon="🔄",
    layout="wide",
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
# GENERAL UTILITIES
# ============================================================

def normalize_columns(df):

    output = df.copy()

    output.columns = [
        str(column).strip()
        for column in output.columns
    ]

    return output


def validate_dataset(
    df,
    country_col="geoUnit"
):

    if country_col not in df.columns:

        raise ValueError(
            f"Required country column '{country_col}' "
            f"was not found."
        )

    missing_year_columns = [
        year
        for year in AFRICA_YEARS
        if year not in df.columns
    ]

    if missing_year_columns:

        raise ValueError(
            "The uploaded Excel file is missing the "
            "following required year columns: "
            + ", ".join(missing_year_columns)
        )


def make_numeric_year_data(
    df,
    year_cols
):

    output = df.copy()

    for column in year_cols:

        output[column] = pd.to_numeric(
            output[column],
            errors="coerce"
        )

    return output


# ============================================================
# STEP 2 — DETECT COMPLETELY MISSING AFRICAN COUNTRIES
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
# STEP 4 — TRAIN COUNTRY-LEVEL WORLD MODEL
# ============================================================

def train_country_world_model(
    df,
    country_col="geoUnit",
    year_cols=None,
    n_estimators=500
):

    """
    Train a temporal world model.

    State:

        [t-3, t-2, t-1]

    Target:

        t

    Example:

        2015, 2016, 2017 -> 2018
        2016, 2017, 2018 -> 2019
        etc.
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    data = make_numeric_year_data(
        df,
        year_cols
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
            dtype=float,
            copy=True
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
            "complete training samples are required."
        )

    model = RandomForestRegressor(

        n_estimators=int(
            n_estimators
        ),

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
# SAFE MODEL PREDICTION
# ============================================================

def safe_predict(
    model,
    state
):

    state = np.asarray(
        state,
        dtype=float
    )

    if state.shape != (3,):

        return np.nan

    if np.isnan(state).any():

        return np.nan

    prediction = model.predict(
        state.reshape(1, -1)
    )

    prediction = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return prediction


# ============================================================
# GLOBAL MEAN
# ============================================================

def robust_global_mean(
    data,
    year_cols
):

    values = data[
        year_cols
    ].to_numpy(
        dtype=float
    )

    value = np.nanmean(
        values
    )

    if np.isfinite(value):

        return float(value)

    return 0.0


# ============================================================
# NEAREST PREVIOUS VALUE
# ============================================================

def get_nearest_left(
    values,
    j
):

    for k in range(
        j - 1,
        -1,
        -1
    ):

        if np.isfinite(
            values[k]
        ):

            return float(
                values[k]
            )

    return np.nan


# ============================================================
# NEAREST NEXT VALUE
# ============================================================

def get_nearest_right(
    values,
    j
):

    for k in range(
        j + 1,
        len(values)
    ):

        if np.isfinite(
            values[k]
        ):

            return float(
                values[k]
            )

    return np.nan


# ============================================================
# NEIGHBOUR PREDICTION
# ============================================================

def neighbour_prediction(
    values,
    j
):

    left = get_nearest_left(
        values,
        j
    )

    right = get_nearest_right(
        values,
        j
    )

    if (
        np.isfinite(left)
        and
        np.isfinite(right)
    ):

        return (
            left + right
        ) / 2.0

    if np.isfinite(left):

        return left

    if np.isfinite(right):

        return right

    return np.nan


# ============================================================
# BUILD MODEL PREDICTION
# ============================================================

def build_prediction(
    values,
    j,
    world_model,
    global_mean
):

    prediction = np.nan

    # --------------------------------------------------------
    # 1. WORLD MODEL
    # --------------------------------------------------------

    if j >= 3:

        state = values[
            j - 3:j
        ]

        if not np.isnan(
            state
        ).any():

            prediction = safe_predict(
                world_model,
                state
            )

    # --------------------------------------------------------
    # 2. NEIGHBOUR FALLBACK
    # --------------------------------------------------------

    if not np.isfinite(
        prediction
    ):

        prediction = neighbour_prediction(
            values,
            j
        )

    # --------------------------------------------------------
    # 3. ROW MEAN FALLBACK
    # --------------------------------------------------------

    if not np.isfinite(
        prediction
    ):

        row_mean = np.nanmean(
            values
        )

        if np.isfinite(
            row_mean
        ):

            prediction = float(
                row_mean
            )

    # --------------------------------------------------------
    # 4. GLOBAL MEAN FALLBACK
    # --------------------------------------------------------

    if not np.isfinite(
        prediction
    ):

        prediction = float(
            global_mean
        )

    return float(
        prediction
    )


# ============================================================
# STEP 5A — IMPUTE EXISTING COUNTRIES
# ============================================================

def impute_existing_country_missing_values(
    df,
    world_model,
    year_cols=None,
    alpha=0.08,
    episodes=200
):

    """
    Fill only originally missing values.

    Observed values are NEVER modified.
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    work = make_numeric_year_data(
        df,
        year_cols
    )

    # --------------------------------------------------------
    # ORIGINAL MISSING MASK
    # --------------------------------------------------------

    original_missing = (
        work[
            year_cols
        ]
        .isna()
        .copy()
    )

    global_mean = robust_global_mean(
        work,
        year_cols
    )

    imputed_count = 0

    # --------------------------------------------------------
    # PROCESS EACH COUNTRY
    # --------------------------------------------------------

    for idx in work.index:

        # IMPORTANT:
        # copy=True makes the NumPy array writable.
        values = (
            work.loc[
                idx,
                year_cols
            ]
            .to_numpy(
                dtype=float,
                copy=True
            )
        )

        mask = (
            original_missing
            .loc[
                idx,
                year_cols
            ]
            .to_numpy(
                dtype=bool,
                copy=True
            )
        )

        if not mask.any():

            continue

        # ----------------------------------------------------
        # MODEL-BASED RL + ONLINE OPTIMIZATION
        # ----------------------------------------------------

        for _ in range(
            int(episodes)
        ):

            previous = values.copy()

            for j in range(
                len(values)
            ):

                # NEVER MODIFY OBSERVED VALUES
                if not mask[j]:

                    continue

                prediction = build_prediction(

                    values=values,

                    j=j,

                    world_model=world_model,

                    global_mean=global_mean
                )

                if not np.isfinite(
                    prediction
                ):

                    continue

                # ------------------------------------------------
                # INITIAL MODEL-BASED UPDATE
                # ------------------------------------------------

                if not np.isfinite(
                    values[j]
                ):

                    values[j] = prediction

                # ------------------------------------------------
                # ONLINE OPTIMIZATION UPDATE
                # ------------------------------------------------

                else:

                    values[j] = (

                        values[j]

                        +

                        float(alpha)
                        *
                        (
                            prediction
                            -
                            values[j]
                        )

                    )

            # ----------------------------------------------------
            # CONVERGENCE
            # ----------------------------------------------------

            finite_mask = np.isfinite(
                values
            )

            if finite_mask.any():

                difference = np.max(

                    np.abs(

                        values[
                            finite_mask
                        ]

                        -

                        previous[
                            finite_mask
                        ]

                    )

                )

            else:

                difference = 0.0

            if difference < 1e-6:

                break

        # ----------------------------------------------------
        # WRITE BACK
        # ----------------------------------------------------

        work.loc[
            idx,
            year_cols
        ] = np.round(
            values,
            3
        )

        imputed_count += int(
            mask.sum()
        )

    return (
        work,
        imputed_count
    )


# ============================================================
# COUNTRY-SPECIFIC RANDOM GENERATOR
# ============================================================

def stable_country_rng(
    country_code,
    random_seed=42
):

    """
    Creates a reproducible random generator for each
    African ISO-3 country.

    Therefore:

        ZAF != NGA != AGO

    rather than every completely missing country receiving
    exactly the same initial values.
    """

    key = (
        f"{country_code}|{random_seed}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        key
    ).hexdigest()

    seed = int(
        digest[:16],
        16
    ) % (
        2**32 - 1
    )

    return np.random.default_rng(
        seed
    )


# ============================================================
# ESTIMATE INITIAL STATE
# ============================================================

def estimate_country_initial_state(
    reference_df,
    year_cols,
    country_code,
    random_seed=42
):

    """
    Estimate 2015, 2016 and 2017 for a completely
    missing country.

    Empirical bootstrap + small stochastic perturbation.

    This prevents all completely missing countries from
    receiving identical initial values.
    """

    data = make_numeric_year_data(
        reference_df,
        year_cols
    )

    rng = stable_country_rng(
        country_code,
        random_seed
    )

    initial_values = []

    # --------------------------------------------------------
    # 2015, 2016, 2017
    # --------------------------------------------------------

    for year in year_cols[:3]:

        series = (
            data[year]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        series = series[
            np.isfinite(series)
        ]

        if len(series) == 0:

            initial_values.append(
                np.nan
            )

            continue

        # ----------------------------------------------------
        # COUNTRY-SPECIFIC EMPIRICAL BOOTSTRAP
        # ----------------------------------------------------

        value = float(
            rng.choice(
                series
            )
        )

        # ----------------------------------------------------
        # SMALL COUNTRY-SPECIFIC PERTURBATION
        # ----------------------------------------------------

        if len(series) >= 3:

            std = float(
                np.nanstd(
                    series
                )
            )

            if (
                np.isfinite(std)
                and
                std > 0
            ):

                value += float(

                    rng.normal(
                        0.0,
                        std * 0.02
                    )

                )

        initial_values.append(
            value
        )

    # --------------------------------------------------------
    # GLOBAL FALLBACK
    # --------------------------------------------------------

    global_mean = robust_global_mean(
        data,
        year_cols
    )

    initial_values = [

        global_mean
        if not np.isfinite(value)
        else value

        for value
        in initial_values

    ]

    return np.asarray(
        initial_values,
        dtype=float
    )


# ============================================================
# STEP 5B — GENERATE COMPLETELY MISSING COUNTRY
# ============================================================

def generate_missing_country_trajectory(
    world_model,
    reference_df,
    year_cols,
    country_code,
    alpha=0.08,
    episodes=200,
    random_seed=42
):

    """
    Generate complete 2015-2025 trajectory for a country
    that is completely absent from the uploaded dataset.
    """

    data = make_numeric_year_data(
        reference_df,
        year_cols
    )

    # --------------------------------------------------------
    # CREATE WRITABLE ARRAY
    # --------------------------------------------------------

    values = np.full(
        len(year_cols),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    values[:3] = (
        estimate_country_initial_state(

            reference_df=data,

            year_cols=year_cols,

            country_code=country_code,

            random_seed=random_seed

        )
    )

    global_mean = robust_global_mean(
        data,
        year_cols
    )

    # --------------------------------------------------------
    # GENERATE 2018-2025
    # --------------------------------------------------------

    for j in range(
        3,
        len(year_cols)
    ):

        state = values[
            j - 3:j
        ]

        prediction = safe_predict(
            world_model,
            state
        )

        if not np.isfinite(
            prediction
        ):

            prediction = global_mean

        values[j] = prediction

    # --------------------------------------------------------
    # ONLINE OPTIMIZATION
    # --------------------------------------------------------

    for _ in range(
        int(episodes)
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

            prediction = safe_predict(
                world_model,
                state
            )

            if not np.isfinite(
                prediction
            ):

                continue

            values[j] = (

                values[j]

                +

                float(alpha)
                *
                (
                    prediction
                    -
                    values[j]
                )

            )

        difference = np.max(

            np.abs(
                values
                -
                previous
            )

        )

        if difference < 1e-6:

            break

    return np.round(
        values,
        3
    )


# ============================================================
# ADD COMPLETELY MISSING AFRICAN COUNTRIES
# ============================================================

def add_missing_african_countries(
    df,
    missing_countries,
    world_model,
    year_cols=None,
    alpha=0.08,
    episodes=200,
    random_seed=42
):

    """
    Add completely missing African countries.

    Existing countries remain unchanged except for their
    originally missing year values, which should already have
    been imputed.
    """

    if year_cols is None:

        year_cols = AFRICA_YEARS

    result = make_numeric_year_data(
        df,
        year_cols
    ).copy()

    generated_rows = []

    for country in missing_countries:

        trajectory = (
            generate_missing_country_trajectory(

                world_model=world_model,

                reference_df=result,

                year_cols=year_cols,

                country_code=country,

                alpha=alpha,

                episodes=episodes,

                random_seed=random_seed

            )
        )

        row = {
            "geoUnit": country
        }

        for year, value in zip(
            year_cols,
            trajectory
        ):

            row[year] = float(
                value
            )

        generated_rows.append(
            row
        )

    if generated_rows:

        added_df = pd.DataFrame(
            generated_rows
        )

        # ----------------------------------------------------
        # PRESERVE NON-YEAR COLUMNS
        # ----------------------------------------------------

        for column in result.columns:

            if column not in added_df.columns:

                added_df[column] = np.nan

        # Ensure exactly the same column order.
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
# COUNT MISSING VALUES
# ============================================================

def count_missing_values(
    df,
    year_cols
):

    numeric = make_numeric_year_data(
        df,
        year_cols
    )

    return int(
        numeric[
            year_cols
        ]
        .isna()
        .sum()
        .sum()
    )


# ============================================================
# BEFORE / AFTER
# ============================================================

def compare_missing_before_after(
    before_df,
    after_df,
    year_cols
):

    before = int(

        make_numeric_year_data(
            before_df,
            year_cols
        )[year_cols]
        .isna()
        .sum()
        .sum()

    )

    after = int(

        make_numeric_year_data(
            after_df,
            year_cols
        )[year_cols]
        .isna()
        .sum()
        .sum()

    )

    return before, after


# ============================================================
# TRAJECTORY DIAGNOSTICS
# ============================================================

def trajectory_variation_table(
    df,
    countries,
    year_cols
):

    subset = df[
        df["geoUnit"]
        .astype(str)
        .isin(countries)
    ][
        ["geoUnit"] + year_cols
    ].copy()

    if subset.empty:

        return subset

    subset[
        "Unique trajectory values"
    ] = subset[
        year_cols
    ].nunique(
        axis=1
    )

    return subset


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🔄 Model-Based RL + Online Optimization Imputer"
)

st.markdown(
    """
This application performs **Model-Based Reinforcement Learning +
Online Optimization** for African country data.

### Processing stages

**Stage 1**
Existing African countries are retained and only their missing
2015–2025 values are imputed.

**Stage 2**
Completely absent African ISO-3 countries are detected and added.

**Stage 3**
A country-level temporal world model generates their missing
trajectories.

**Stage 4**
Online optimization iteratively refines the generated values.

> Completely absent countries are model-generated estimates, not
> observed values.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Settings"
)

st.sidebar.number_input(
    "Start year",
    min_value=2015,
    max_value=2015,
    value=2015
)

st.sidebar.number_input(
    "End year",
    min_value=2025,
    max_value=2025,
    value=2025
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

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1
)

st.sidebar.info(
    "African coverage: 2015–2025."
)


# ============================================================
# EXCEL UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Excel dataset",
    type=[
        "xlsx",
        "xls"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload an Excel file containing a 'geoUnit' "
        "column and year columns 2015–2025."
    )

    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

try:

    df = pd.read_excel(
        uploaded_file
    )

    df = normalize_columns(
        df
    )

    validate_dataset(
        df,
        country_col="geoUnit"
    )

except Exception as e:

    st.error(
        "The application encountered an error "
        "while reading the uploaded dataset."
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# DATASET SUMMARY
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
        "Missing year values",
        count_missing_values(
            df,
            AFRICA_YEARS
        )
    )


with st.expander(
    "View uploaded data"
):

    st.dataframe(
        df,
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


existing_africa = set(

    df["geoUnit"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()

)


existing_africa_count = len(

    existing_africa
    &
    ALL_AFRICA

)


c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "African countries expected",
        len(ALL_AFRICA)
    )

with c2:

    st.metric(
        "African countries present",
        existing_africa_count
    )

with c3:

    st.metric(
        "Completely missing",
        len(missing_countries)
    )


if missing_countries:

    st.warning(

        f"{len(missing_countries)} African countries "
        "are completely absent from the uploaded dataset."

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


# ============================================================
# TRAIN COUNTRY WORLD MODEL
# ============================================================

st.subheader(
    "🧠 Country-Level World Model"
)

try:

    with st.spinner(
        "Training Random Forest world model..."
    ):

        world_model, training_samples = (
            train_country_world_model(

                df,

                country_col="geoUnit",

                year_cols=AFRICA_YEARS,

                n_estimators=n_estimators

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


# ============================================================
# RUN BUTTON
# ============================================================

run_button = st.button(

    "🚀 Run Model-Based RL + Online Optimization",

    type="primary",

    use_container_width=True

)


if not run_button:

    st.info(
        "Click the button above to start imputation."
    )

    st.stop()


# ============================================================
# PROGRESS
# ============================================================

st.subheader(
    "🔄 Imputing Missing Years"
)

progress = st.progress(
    0
)

status = st.empty()


# ============================================================
# STEP 1 — EXISTING COUNTRIES
# ============================================================

try:

    status.info(

        "Step 1/2: Imputing missing years "
        "for countries already present..."

    )

    (
        df_existing_imputed,
        existing_imputed_count
    ) = (

        impute_existing_country_missing_values(

            df=df,

            world_model=world_model,

            year_cols=AFRICA_YEARS,

            alpha=alpha,

            episodes=episodes

        )

    )

    progress.progress(
        50
    )

except Exception as e:

    st.error(
        "The application encountered an error "
        "while imputing existing countries."
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# STEP 2 — COMPLETELY MISSING COUNTRIES
# ============================================================

try:

    status.info(

        "Step 2/2: Generating trajectories for "
        "completely missing African countries..."

    )

    df_final = (
        add_missing_african_countries(

            df=df_existing_imputed,

            missing_countries=missing_countries,

            world_model=world_model,

            year_cols=AFRICA_YEARS,

            alpha=alpha,

            episodes=episodes,

            random_seed=int(
                random_seed
            )

        )
    )

    progress.progress(
        100
    )

except Exception as e:

    st.error(
        "The application encountered an error "
        "while adding completely missing countries."
    )

    st.exception(
        e
    )

    st.stop()


status.success(
    "Model-Based RL + Online Optimization completed."
)


# ============================================================
# FINAL RESULTS
# ============================================================

st.subheader(
    "📈 Imputation Results"
)


before_missing, after_missing = (
    compare_missing_before_after(

        df,

        df_final,

        AFRICA_YEARS

    )
)


m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    st.metric(
        "African countries added",
        len(missing_countries)
    )


with m2:

    st.metric(
        "Final number of countries",
        len(df_final)
    )


with m3:

    st.metric(
        "World-model samples",
        training_samples
    )


with m4:

    st.metric(
        "Missing before",
        before_missing
    )


with m5:

    st.metric(
        "Missing after",
        after_missing
    )


# ============================================================
# PRESERVE OBSERVED VALUES CHECK
# ============================================================

st.subheader(
    "🔒 Observed-Value Preservation Check"
)


original_numeric = (
    make_numeric_year_data(
        df,
        AFRICA_YEARS
    )
)


final_numeric = (
    make_numeric_year_data(
        df_final,
        AFRICA_YEARS
    )
)


original_numeric[
    "_country_key"
] = (

    original_numeric[
        "geoUnit"
    ]
    .astype(str)
    .str.strip()
    .str.upper()

)


final_numeric[
    "_country_key"
] = (

    final_numeric[
        "geoUnit"
    ]
    .astype(str)
    .str.strip()
    .str.upper()

)


original_indexed = (

    original_numeric
    .drop_duplicates(
        "_country_key"
    )
    .set_index(
        "_country_key"
    )

)


final_indexed = (

    final_numeric
    .drop_duplicates(
        "_country_key"
    )
    .set_index(
        "_country_key"
    )

)


preserved = True

changed_observed_cells = 0


for country in original_indexed.index:

    if country not in final_indexed.index:

        continue

    for year in AFRICA_YEARS:

        original_value = (
            original_indexed
            .loc[
                country,
                year
            ]
        )

        final_value = (
            final_indexed
            .loc[
                country,
                year
            ]
        )

        if np.isfinite(
            original_value
        ):

            if (

                not np.isfinite(
                    final_value
                )

                or

                not np.isclose(

                    float(
                        original_value
                    ),

                    float(
                        final_value
                    ),

                    rtol=1e-9,

                    atol=1e-9

                )

            ):

                preserved = False

                changed_observed_cells += 1


if preserved:

    st.success(

        "All originally observed year values were preserved. "
        "Only missing values were imputed."

    )

else:

    st.error(

        f"{changed_observed_cells} originally observed "
        "cells appear to have changed."

    )


# ============================================================
# GENERATED AFRICAN COUNTRY TRAJECTORIES
# ============================================================

if missing_countries:

    st.subheader(
        "🌍 Generated Trajectories for Added African Countries"
    )

    generated_table = (
        trajectory_variation_table(

            df_final,

            missing_countries,

            AFRICA_YEARS

        )
    )


    st.dataframe(

        generated_table,

        use_container_width=True

    )


    # --------------------------------------------------------
    # CHECK FOR IDENTICAL TRAJECTORIES
    # --------------------------------------------------------

    duplicate_trajectories = (

        generated_table
        .duplicated(
            subset=AFRICA_YEARS,
            keep=False
        )
        .sum()

    )


    if duplicate_trajectories == 0:

        st.success(

            "Each added African country has a distinct "
            "2015–2025 generated trajectory."

        )

    else:

        st.warning(

            f"{duplicate_trajectories} generated rows share "
            "a complete trajectory."

        )


# ============================================================
# FINAL DATASET
# ============================================================

st.subheader(
    "📋 Final Imputed African Dataset"
)


st.dataframe(

    df_final,

    use_container_width=True

)


# ============================================================
# MISSING VALUES AFTER IMPUTATION
# ============================================================

st.subheader(
    "🔎 Missing Values After Imputation"
)


missing_after_by_year = (

    df_final[
        AFRICA_YEARS
    ]
    .isna()
    .sum()
    .rename(
        "Missing values"
    )
    .to_frame()

)


st.dataframe(

    missing_after_by_year,

    use_container_width=True

)


# ============================================================
# DOWNLOAD EXCEL
# ============================================================

st.subheader(
    "💾 Download Results"
)


output = io.BytesIO()


with pd.ExcelWriter(

    output,

    engine="openpyxl"

) as writer:

    # --------------------------------------------------------
    # FINAL DATA
    # --------------------------------------------------------

    df_final.to_excel(

        writer,

        index=False,

        sheet_name="Imputed_Africa"

    )

    # --------------------------------------------------------
    # ADDED COUNTRIES
    # --------------------------------------------------------

    pd.DataFrame({

        "Missing African Countries":
        missing_countries

    }).to_excel(

        writer,

        index=False,

        sheet_name="Added_Countries"

    )

    # --------------------------------------------------------
    # MISSING AFTER
    # --------------------------------------------------------

    missing_after_by_year.to_excel(

        writer,

        sheet_name="Missing_After"

    )


output.seek(0)


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
    ),

    use_container_width=True

)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "📚 Methodology used by this application"
):

    st.markdown(
        r"""
## 1. Country-level temporal World Model

The temporal state is defined as:

\[
s_t =
[x_{t-3},x_{t-2},x_{t-1}]
\]

The Random Forest world model learns:

\[
\hat{x}_t =
f_\theta(s_t)
\]

where \(f_\theta\) represents the learned country-level
environment/world model.

For example:

\[
[2015,2016,2017]
\rightarrow
2018
\]

and:

\[
[2016,2017,2018]
\rightarrow
2019
\]

---

## 2. Existing African countries

For countries already present in the uploaded dataset, the
algorithm records the original missing-value mask.

Only cells that were originally missing are updated.

Observed values remain unchanged.

---

## 3. Model-Based RL update

For a missing year, the world model generates:

\[
\hat{x}_t =
f_\theta(s_t)
\]

The estimated value becomes the candidate action/state update.

---

## 4. Online Optimization

The optimization step is:

\[
x_t^{k+1}
=
x_t^k
+
\alpha
\left(
\hat{x}_t-x_t^k
\right)
\]

where:

- \(x_t^k\) = current estimate
- \(\hat{x}_t\) = world-model prediction
- \(\alpha\) = online optimization learning rate

---

## 5. Completely missing African countries

A completely absent country has no observed 2015–2017 values.

Therefore, those values cannot be directly learned from that
country itself.

The application estimates its initial state using empirical
African-country distributions.

Each ISO-3 country receives a deterministic country-specific
random seed.

Consequently, the generated trajectories can differ between
countries instead of every missing country receiving exactly
the same trajectory.

---

## 6. Recursive trajectory generation

Once the country-specific initial state is established:

\[
[2015,2016,2017]
\rightarrow
2018
\]

then:

\[
[2016,2017,2018]
\rightarrow
2019
\]

and so forth until:

\[
2025
\]

---

## 7. Important interpretation

For completely missing countries, the generated observations are
**model estimates**.

They should therefore be labelled as imputed/synthetic values in
a research dataset rather than presented as directly observed
statistics.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Model-Based RL + Online Optimization | "
    "African ISO-3 Coverage | 2015–2025"
)
