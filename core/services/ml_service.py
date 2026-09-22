"""
Kaggle S&P 500 Qlib -> Alpha158 -> LightGBM

FULL VERSION

This script:

    1. Loads Qlib
    2. Loads instruments/all.txt
    3. Creates Alpha158
    4. Creates DatasetH
    5. Uses DataHandlerLP.DK_L
    6. Extracts train/valid/test features
    7. Extracts labels
    8. Validates matrices
    9. Trains LightGBM
   10. Predicts test data
   11. Calculates correlations
   12. Shows feature importance
   13. Saves predictions
   14. Saves feature importance
   15. Creates a proper AAPL diagnostic plot
   16. Creates cross-sectional and AAPL signal diagnostics
   17. Logs experiment to SQLite MLflow


IMPORTANT PLOT FIXES:

The old graph compared:

    prediction ~= 0.02 std

against:

    Alpha158 target ~= 1.0 std

on the same y-axis.

That makes the prediction look like a straight line.

This version uses separate scales and additional plots.

It also DOES NOT cumulatively sum the normalized Alpha158
target as though it were a dollar return.

Instead, it constructs the raw Alpha158 forward return:

    close[t+2] / close[t+1] - 1

and uses the model prediction as a signal for that return.

The strategy curves are diagnostics, not full Qlib execution backtests.
The primary model evaluation is cross-sectional because the Alpha158 label is CS-normalized.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

QLIB_DATA_DIR = PROJECT_ROOT / "qlib_data"

INSTRUMENT_FILE = (
    QLIB_DATA_DIR
    / "instruments"
    / "all.txt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

OUTPUT_DIR = PROJECT_ROOT / "model_output"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "feature_importance.csv"
)

AAPL_PLOT_FILE = (
    OUTPUT_DIR / "aapl_predictions.png"
)

AAPL_SIGNAL_FILE = (
    OUTPUT_DIR / "aapl_signal_diagnostics.png"
)

AAPL_STRATEGY_FILE = (
    OUTPUT_DIR / "aapl_strategy.png"
)


# ============================================================
# DATA RANGE
# ============================================================

DATA_START = "2008-01-01"
DATA_END = "2026-02-20"

TRAIN_START = "2008-01-01"
TRAIN_END = "2014-12-31"

VALID_START = "2015-01-01"
VALID_END = "2016-12-31"

TEST_START = "2017-01-01"
TEST_END = "2026-02-20"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 1000

LEARNING_RATE = 0.03

NUM_LEAVES = 64

MAX_DEPTH = -1

MIN_CHILD_SAMPLES = 100

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

REG_ALPHA = 0.1

REG_LAMBDA = 1.0


# ============================================================
# PLOT CONFIGURATION
# ============================================================

ROLLING_WINDOW = 20

# Signal construction for diagnostics.  Alpha158's default learn processor
# cross-sectionally normalizes the label, so a prediction is fundamentally a
# relative cross-sectional score.  The AAPL-only diagnostic therefore uses
# AAPL's percentile rank among all stocks on the same date instead of treating
# the raw prediction sign as an absolute return forecast.
SIGNAL_LOOKBACK = 60
SIGNAL_MIN_PERIODS = 20
SIGNAL_DEAD_ZONE = 0.15
POSITION_SMOOTHING = 3
POSITION_CLIP = 1.0
TRANSACTION_COST_BPS = 5.0
TRADING_DAYS_PER_YEAR = 252

# Cross-sectional portfolio diagnostic.  The model is trained to rank stocks,
# so this is the primary strategy diagnostic; the AAPL chart remains a useful
# single-name visualization but is not the model's natural objective.
PORTFOLIO_LONG_QUANTILE = 0.20
PORTFOLIO_SHORT_QUANTILE = 0.20
PORTFOLIO_TRANSACTION_COST_BPS = 5.0
PORTFOLIO_MIN_STOCKS = 20

AAPL_STRATEGY_METRICS_FILE = (
    OUTPUT_DIR / "aapl_strategy_metrics.csv"
)

# Small number to avoid division by zero
EPS = 1e-12


# ============================================================
# ALPHA158 CONFIGURATION
# ============================================================

ALPHA158_START = DATA_START

LABEL_EXPRESSION = (
    "Ref($close, -2)/Ref($close, -1) - 1"
)

ALPHA158_END = DATA_END

ALPHA158_FIT_START = TRAIN_START

ALPHA158_FIT_END = TRAIN_END


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

def check_environment():

    print_header(
        "CHECKING ENVIRONMENT"
    )

    print(
        f"\nPython:\n    {sys.executable}"
    )

    print(
        f"\nProject root:\n    {PROJECT_ROOT}"
    )

    print(
        f"\nQlib data:\n    {QLIB_DATA_DIR}"
    )

    print(
        f"\nInstrument file:\n    {INSTRUMENT_FILE}"
    )

    print(
        f"\nMLflow database:\n    {MLFLOW_DB}"
    )

    if not QLIB_DATA_DIR.exists():

        raise FileNotFoundError(
            f"""
Qlib data directory does not exist:

    {QLIB_DATA_DIR}
"""
        )

    if not INSTRUMENT_FILE.exists():

        raise FileNotFoundError(
            f"""
Qlib instrument file does not exist:

    {INSTRUMENT_FILE}
"""
        )


# ============================================================
# LOAD INSTRUMENTS
# ============================================================

def load_instruments():

    print_header(
        "LOADING QLIB INSTRUMENTS"
    )

    print(
        f"\nInstrument file:\n    {INSTRUMENT_FILE}"
    )

    instruments = pd.read_csv(
        INSTRUMENT_FILE,
        sep="\t",
        header=None,
        names=[
            "instrument",
            "start",
            "end",
        ],
    )

    instruments["instrument"] = (
        instruments["instrument"]
        .astype(str)
        .str.strip()
    )

    instruments = instruments[
        instruments["instrument"] != ""
    ]

    instruments = instruments.reset_index(
        drop=True
    )

    if instruments.empty:

        raise RuntimeError(
            "all.txt exists but contains no instruments."
        )

    print(
        f"\nNumber of instruments:\n"
        f"    {len(instruments):,}"
    )

    print(
        "\nFirst instruments:"
    )

    for symbol in instruments[
        "instrument"
    ].head(20):

        print(
            f"    {symbol}"
        )

    return instruments


# ============================================================
# INITIALIZE QLIB
# ============================================================

def initialize_qlib():

    print_header(
        "INITIALIZING QLIB"
    )

    import qlib

    from qlib.config import REG_US

    mlflow_uri = (
        f"sqlite:///{MLFLOW_DB.resolve()}"
    )

    print(
        f"""
Project root:

    {PROJECT_ROOT}

Qlib data:

    {QLIB_DATA_DIR}

MLflow database:

    {MLFLOW_DB}

MLflow URI:

    {mlflow_uri}
"""
    )

    qlib.init(
        provider_uri=str(
            QLIB_DATA_DIR.resolve()
        ),
        region=REG_US,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": mlflow_uri,
                "default_exp_name":
                    "Alpha158_LightGBM",
            },
        },
    )

    print(
        "\nQlib initialized successfully."
    )

    return qlib


# ============================================================
# CREATE ALPHA158
# ============================================================

def create_alpha158(instruments):

    print_header(
        "CREATING ALPHA158 HANDLER"
    )

    from qlib.contrib.data.handler import Alpha158

    print(
        f"""
Date range:

    {ALPHA158_START}
    ->
    {ALPHA158_END}

Fit range:

    {ALPHA158_FIT_START}
    ->
    {ALPHA158_FIT_END}

Number of instruments:

    {len(instruments):,}

Label expression:

    {LABEL_EXPRESSION}
"""
    )

    instrument_list = (
        instruments["instrument"].tolist()
    )

    # Alpha158 accepts the label configuration through **kwargs.
    # Pass it explicitly instead of relying on the class default so the
    # training target is tied to the label used by the rest of this script.
    handler = Alpha158(
        instruments=instrument_list,
        start_time=ALPHA158_START,
        end_time=ALPHA158_END,
        fit_start_time=ALPHA158_FIT_START,
        fit_end_time=ALPHA158_FIT_END,
        label=[LABEL_EXPRESSION],
        drop_raw=False,
    )

    print("\nAlpha158 handler created.")

    # Verify the handler's actual label configuration immediately.  This
    # catches accidental changes in Qlib configuration before model training.
    try:
        label_config = handler.get_label_config()
        print("\n========== ALPHA158 LABEL CONFIG ==========")
        print(label_config)

        configured_expressions = label_config[0]
        if configured_expressions != [LABEL_EXPRESSION]:
            raise RuntimeError(
                "Alpha158 label configuration does not match "
                f"LABEL_EXPRESSION={LABEL_EXPRESSION!r}.\n"
                f"Actual configuration: {label_config!r}"
            )
    except AttributeError:
        print(
            "\nWARNING: this Qlib Alpha158 implementation does not "
            "expose get_label_config(). The explicit label argument was "
            "still passed to the handler."
        )

    print("\n========== LEARN PROCESSORS ==========")
    print(getattr(handler, "learn_processors", "<unavailable>"))

    print("\n========== INFER PROCESSORS ==========")
    print(getattr(handler, "infer_processors", "<unavailable>"))

    return handler


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(handler):

    print_header(
        "CREATING DATASET"
    )

    from qlib.data.dataset import DatasetH

    segments = {
        "train": (
            TRAIN_START,
            TRAIN_END,
        ),
        "valid": (
            VALID_START,
            VALID_END,
        ),
        "test": (
            TEST_START,
            TEST_END,
        ),
    }

    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )

    print(
        "\nDataset created."
    )

    return dataset


# ============================================================
# FLATTEN COLUMNS
# ============================================================

def flatten_feature_columns(df):

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        new_columns = []

        for column in df.columns:

            if isinstance(
                column,
                tuple,
            ):

                if len(column) >= 2:

                    new_columns.append(
                        str(column[-1])
                    )

                else:

                    new_columns.append(
                        str(column[0])
                    )

            else:

                new_columns.append(
                    str(column)
                )

        df = df.copy()

        df.columns = new_columns

    else:

        df = df.copy()

        df.columns = [
            str(column)
            for column in df.columns
        ]

    return df


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(
    dataset,
    segment,
    data_key,
):

    print_subheader(
        f"EXTRACTING {segment.upper()} FEATURES"
    )

    df = dataset.prepare(
        segment,
        col_set="feature",
        data_key=data_key,
    )

    if isinstance(
        df,
        list,
    ):

        if len(df) != 1:

            raise RuntimeError(
                f"""
Expected one DataFrame for segment
'{segment}', but received {len(df)}.
"""
            )

        df = df[0]

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            f"""
Expected pandas DataFrame.

Received:

    {type(df)}
"""
        )

    df = flatten_feature_columns(
        df
    )

    if df.ndim != 2:

        raise ValueError(
            f"""
Feature matrix is not 2-dimensional.

Shape:

    {df.shape}
"""
        )

    if df.empty:

        raise ValueError(
            f"""
Feature matrix for '{segment}' is empty.
"""
        )

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    if df.columns.duplicated().any():

        duplicate_columns = (
            df.columns[
                df.columns.duplicated()
            ]
            .tolist()
        )

        print(
            "\nWARNING: duplicate feature names:"
        )

        print(
            duplicate_columns
        )

        df = df.loc[
            :,
            ~df.columns.duplicated(),
        ]

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # NaN check
    # --------------------------------------------------------

    nan_count = int(
        df.isna()
        .sum()
        .sum()
    )

    print(
        f"\nShape:\n"
        f"    {df.shape}"
    )

    print(
        f"\nRows:\n"
        f"    {len(df):,}"
    )

    print(
        f"\nFeatures:\n"
        f"    {len(df.columns):,}"
    )

    print(
        f"\nMissing values:\n"
        f"    {nan_count:,}"
    )

    if nan_count > 0:

        print(
            "\nWARNING:"
        )

        print(
            "NaNs found in feature matrix."
        )

        print(
            "Replacing remaining NaNs with 0."
        )

        df = df.fillna(0.0)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"""
Feature matrix became empty.

Segment:

    {segment}
"""
        )

    if len(df.columns) == 0:

        raise ValueError(
            f"""
Feature matrix has zero columns.

Segment:

    {segment}
"""
        )

    print(
        "\nFeature extraction successful."
    )

    return df


# ============================================================
# EXTRACT LABELS
# ============================================================

def extract_labels(
    dataset,
    segment,
    data_key,
):

    print_subheader(
        f"EXTRACTING {segment.upper()} LABELS"
    )

    labels = dataset.prepare(
        segment,
        col_set="label",
        data_key=data_key,
    )

    if isinstance(
        labels,
        list,
    ):

        if len(labels) != 1:

            raise RuntimeError(
                f"""
Expected one DataFrame for labels,
received {len(labels)}.
"""
            )

        labels = labels[0]

    if not isinstance(
        labels,
        pd.DataFrame,
    ):

        raise TypeError(
            f"""
Expected pandas DataFrame for labels.

Received:

    {type(labels)}
"""
        )

    labels = flatten_feature_columns(
        labels
    )

    if labels.empty:

        raise ValueError(
            f"""
Label DataFrame for '{segment}' is empty.
"""
        )

    if len(labels.columns) == 0:

        raise ValueError(
            "No label columns found."
        )

    print(
        f"\nLabel shape:\n"
        f"    {labels.shape}"
    )

    print(
        f"\nLabel columns:\n"
        f"    {labels.columns.tolist()}"
    )

    # --------------------------------------------------------
    # Use first label
    # --------------------------------------------------------

    if len(labels.columns) > 1:

        print(
            "\nWARNING:"
        )

        print(
            "More than one label column found."
        )

        print(
            f"Using: {labels.columns[0]}"
        )

        labels = labels.iloc[:, [0]]

    label_column = labels.columns[0]

    y = pd.to_numeric(
        labels[label_column],
        errors="coerce",
    )

    y.name = "label"

    y = y.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    before = len(y)

    y = y.loc[
        y.notna()
    ]

    removed = (
        before - len(y)
    )

    if removed > 0:

        print(
            f"\nRemoved {removed:,} "
            "rows with invalid labels."
        )

    if y.empty:

        raise ValueError(
            f"""
All labels for '{segment}' are invalid.
"""
        )

    print(
        f"\nValid labels:\n"
        f"    {len(y):,}"
    )

    print(
        "\nLabel statistics:"
    )

    print(
        y.describe()
    )

    return y


# ============================================================
# ALIGN FEATURES AND LABELS
# ============================================================

def align_features_labels(
    X,
    y,
    segment,
):

    print_subheader(
        f"ALIGNING {segment.upper()} "
        "FEATURES + LABELS"
    )

    print(
        f"""
Features before alignment:

    {X.shape}

Labels before alignment:

    {y.shape}
"""
    )

    common_index = (
        X.index.intersection(
            y.index
        )
    )

    if len(common_index) == 0:

        raise ValueError(
            f"""
No common observations for '{segment}'.
"""
        )

    X = X.loc[
        common_index
    ]

    y = y.loc[
        common_index
    ]

    X = X.sort_index()

    y = y.reindex(
        X.index
    )

    if len(X) != len(y):

        raise RuntimeError(
            f"""
Feature/label length mismatch.

X:

    {len(X)}

y:

    {len(y)}
"""
        )

    print(
        f"""
Features after alignment:

    {X.shape}

Labels after alignment:

    {y.shape}

Alignment successful.
"""
    )

    return X, y


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    dataset,
    data_key,
):

    print_header(
        "PREPARING TRAIN / VALID / TEST DATA"
    )

    X_train = extract_features(
        dataset,
        "train",
        data_key,
    )

    y_train = extract_labels(
        dataset,
        "train",
        data_key,
    )

    X_train, y_train = (
        align_features_labels(
            X_train,
            y_train,
            "train",
        )
    )

    X_valid = extract_features(
        dataset,
        "valid",
        data_key,
    )

    y_valid = extract_labels(
        dataset,
        "valid",
        data_key,
    )

    X_valid, y_valid = (
        align_features_labels(
            X_valid,
            y_valid,
            "valid",
        )
    )

    X_test = extract_features(
        dataset,
        "test",
        data_key,
    )

    y_test = extract_labels(
        dataset,
        "test",
        data_key,
    )

    X_test, y_test = (
        align_features_labels(
            X_test,
            y_test,
            "test",
        )
    )

    # --------------------------------------------------------
    # Feature consistency
    # --------------------------------------------------------

    print_subheader(
        "CHECKING FEATURE COLUMN CONSISTENCY"
    )

    train_columns = list(
        X_train.columns
    )

    valid_columns = list(
        X_valid.columns
    )

    test_columns = list(
        X_test.columns
    )

    if train_columns != valid_columns:

        raise ValueError(
            "Train/validation feature mismatch."
        )

    if train_columns != test_columns:

        raise ValueError(
            "Train/test feature mismatch."
        )

    print(
        "\nFeature columns match across all segments."
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print_subheader(
        "FINAL LIGHTGBM INPUT VALIDATION"
    )

    datasets = [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]

    for name, X, y in datasets:

        print(
            f"""
{name.upper()}:

    X shape: {X.shape}
    y shape: {y.shape}
    X ndim:  {X.ndim}
    y ndim:  {y.ndim}
"""
        )

        if X.ndim != 2:

            raise ValueError(
                f"{name} X is not 2-dimensional."
            )

        if X.empty:

            raise ValueError(
                f"{name} X is empty."
            )

        if y.empty:

            raise ValueError(
                f"{name} y is empty."
            )

        if len(X) != len(y):

            raise ValueError(
                f"{name} X/y length mismatch."
            )

    print(
        "\nAll LightGBM inputs are valid."
    )

    # Sanity-check the target before the expensive model fit.  The default
    # Alpha158 learn processor applies CSZScoreNorm to labels, so the model
    # target is intentionally a normalized cross-sectional signal rather
    # than a raw percentage return.
    for name, y in (
        ("train", y_train),
        ("valid", y_valid),
        ("test", y_test),
    ):
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{name} labels contain non-finite values after cleaning."
            )

    print("\nTarget sanity check passed.")

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )


# ============================================================
# CHECK ALPHA158
# ============================================================

def inspect_alpha158_features(
    X_train,
):

    print_header(
        "CHECKING ALPHA158 FEATURES"
    )

    print(
        "\nTraining feature shape:"
    )

    print(
        X_train.shape
    )

    print(
        f"""
Number of observations:

    {len(X_train):,}

Number of features:

    {len(X_train.columns):,}

Missing values:

    {int(
        X_train.isna()
        .sum()
        .sum()
    ):,}
"""
    )

    print(
        "\nCalculating unique values per feature..."
    )

    unique_counts = (
        X_train
        .nunique(
            dropna=False
        )
        .sort_values()
    )

    print(
        "\nUnique-value statistics:"
    )

    print(
        unique_counts.describe()
    )

    low_cardinality = (
        unique_counts[
            unique_counts <= 5
        ]
    )

    print(
        "\nFeatures with 5 or fewer unique values:"
    )

    if low_cardinality.empty:

        print("    None")

    else:

        print(
            low_cardinality
        )

    constant_features = (
        unique_counts[
            unique_counts <= 1
        ]
    )

    print(
        "\nConstant features:"
    )

    if constant_features.empty:

        print("    None")

    else:

        for feature in (
            constant_features.index
        ):

            print(
                f"    {feature}"
            )

    print(
        "\nFeature standard deviation:"
    )

    stds = X_train.std()

    print(
        stds.describe()
    )

    zero_std = (
        stds[
            stds == 0
        ]
    )

    print(
        "\nFeatures with zero standard deviation:"
    )

    if zero_std.empty:

        print("    None")

    else:

        print(
            zero_std
        )

    print(
        "\nFirst 5 rows:"
    )

    print(
        X_train.head()
    )


# ============================================================
# CREATE LIGHTGBM
# ============================================================

def create_lightgbm():

    print_header(
        "CREATING LIGHTGBM MODEL"
    )

    import lightgbm as lgb

    print(
        f"""
Parameters:

    objective:
        regression

    n_estimators:
        {N_ESTIMATORS}

    learning_rate:
        {LEARNING_RATE}

    num_leaves:
        {NUM_LEAVES}

    max_depth:
        {MAX_DEPTH}

    min_child_samples:
        {MIN_CHILD_SAMPLES}

    subsample:
        {SUBSAMPLE}

    colsample_bytree:
        {COLSAMPLE_BYTREE}

    reg_alpha:
        {REG_ALPHA}

    reg_lambda:
        {REG_LAMBDA}

    random_state:
        {RANDOM_STATE}
"""
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=RANDOM_STATE,
        bagging_freq=1,
        bagging_seed=RANDOM_STATE,
        feature_fraction_seed=RANDOM_STATE,
        data_random_seed=RANDOM_STATE,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    print_header(
        "TRAINING"
    )

    print(
        f"""
Training observations:

    {len(X_train):,}

Validation observations:

    {len(X_valid):,}

Features:

    {len(X_train.columns):,}
"""
    )

    print(
        "\nTraining LightGBM..."
    )

    import lightgbm as lgb

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_train,
                y_train,
            ),
            (
                X_valid,
                y_valid,
            ),
        ],
        eval_names=[
            "train",
            "valid",
        ],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=20
            ),
        ],
    )

    print(
        "\nTraining finished."
    )

    best_iteration = getattr(
        model,
        "best_iteration_",
        None,
    )

    print(
        f"""
Best iteration:

    {best_iteration if best_iteration is not None else "N/A"}
"""
    )

    # Report validation correlation immediately so a run that completes but
    # produces a weak signal is obvious before the later plotting stage.
    valid_prediction = pd.Series(
        model.predict(X_valid),
        index=y_valid.index,
    )
    valid_corr = valid_prediction.corr(y_valid)
    print(
        "\nValidation prediction / label correlation:"
    )
    print(
        f"    {valid_corr:.6f}"
        if pd.notna(valid_corr)
        else "    NaN"
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X_test,
):

    print_header(
        "PREDICTING"
    )

    print(
        f"""
Prediction input:

    rows:
        {len(X_test):,}

    features:
        {len(X_test.columns):,}

    shape:
        {X_test.shape}
"""
    )

    prediction = model.predict(
        X_test
    )

    prediction = pd.Series(
        prediction,
        index=X_test.index,
        name="prediction",
        dtype=float,
    )

    print(
        "\nPrediction statistics:"
    )

    print(
        prediction.describe()
    )

    print(
        "\nUnique predictions:"
    )

    print(
        prediction.nunique()
    )

    print(
        "\nMost common predictions:"
    )

    print(
        prediction.value_counts(
            dropna=False
        ).head(20)
    )

    return prediction


# ============================================================
# BUILD RESULTS
# ============================================================

def build_results(
    prediction,
    y_test,
):

    print_header(
        "GETTING TEST LABELS"
    )

    results = pd.concat(
        [
            prediction,
            y_test.rename(
                "actual"
            ),
        ],
        axis=1,
        join="inner",
    )

    results = results.dropna()

    if results.empty:

        raise RuntimeError(
            "Prediction/actual DataFrame is empty."
        )

    print(
        "\nResults:"
    )

    print(
        results.head()
    )

    correlation = (
        results[
            "prediction"
        ]
        .corr(
            results["actual"]
        )
    )

    print(
        "\nPrediction / actual correlation:"
    )

    print(
        f"    {correlation:.6f}"
    )

    return results, correlation


# ============================================================
# CHECK AAPL
# ============================================================

def check_aapl(
    results,
):

    print_header(
        "CHECKING AAPL"
    )

    if not isinstance(
        results.index,
        pd.MultiIndex,
    ):

        print(
            "\nResults index is not MultiIndex."
        )

        return None

    if "instrument" not in (
        results.index.names
    ):

        print(
            "\nCould not find instrument index level."
        )

        return None

    try:

        aapl = results.xs(
            "AAPL",
            level="instrument",
        )

    except KeyError:

        print(
            "\nAAPL is not present in test results."
        )

        return None

    aapl = aapl.copy()

    aapl.index = pd.to_datetime(
        aapl.index
    )

    aapl = aapl.sort_index()

    print(
        "\nAAPL sample:"
    )

    print(
        aapl.head(20)
    )

    if len(aapl) >= 2:

        correlation = (
            aapl[
                "prediction"
            ]
            .corr(
                aapl["actual"]
            )
        )

        print(
            "\nAAPL correlation:"
        )

        print(
            f"    {correlation:.6f}"
        )

    return aapl


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance(
    model,
    feature_names,
):

    print_header(
        "FEATURE IMPORTANCE"
    )

    importance = (
        model.feature_importances_
    )

    if len(importance) != len(
        feature_names
    ):

        raise RuntimeError(
            """
Feature importance length does not
match feature count.
"""
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nFeature importance array:"
    )

    print(
        importance
    )

    print(
        "\nTop features:"
    )

    for _, row in (
        importance_df
        .head(20)
        .iterrows()
    ):

        print(
            f"    {row['feature']}: "
            f"{int(row['importance'])}"
        )

    return importance_df


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    results,
    importance_df,
):

    print_header(
        "SAVING OUTPUTS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        PREDICTIONS_FILE
    )

    print(
        f"""
Predictions saved:

    {PREDICTIONS_FILE}
"""
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"""
Feature importance saved:

    {FEATURE_IMPORTANCE_FILE}
"""
    )


# ============================================================
# LOAD AAPL PRICE
# ============================================================

def load_aapl_price():

    print_header(
        "LOADING AAPL RAW PRICE DATA"
    )

    try:

        from qlib.data import D

        price = D.features(
            ["AAPL"],
            [
                "$close",
            ],
            start_time=TEST_START,
            end_time=TEST_END,
            freq="day",
        )

    except Exception as exc:

        print(
            "\nCould not load AAPL price data:"
        )

        print(
            f"    {exc}"
        )

        return None

    if price is None:

        print(
            "\nQlib returned None."
        )

        return None

    if price.empty:

        print(
            "\nAAPL price data is empty."
        )

        return None

    # --------------------------------------------------------
    # Flatten columns
    # --------------------------------------------------------

    if isinstance(
        price.columns,
        pd.MultiIndex,
    ):

        price.columns = [
            str(c[-1])
            for c in price.columns
        ]

    else:

        price.columns = [
            str(c)
            for c in price.columns
        ]

    if "$close" not in (
        price.columns
    ):

        print(
            "\n$close not found."
        )

        print(
            price.columns.tolist()
        )

        return None

    price = price[
        ["$close"]
    ].copy()

    price = price.rename(
        columns={
            "$close": "close"
        }
    )

    # --------------------------------------------------------
    # Remove AAPL instrument level
    # --------------------------------------------------------

    if isinstance(
        price.index,
        pd.MultiIndex,
    ):

        try:

            price = price.xs(
                "AAPL",
                level="instrument",
            )

        except Exception:

            try:

                price = price.xs(
                    "AAPL",
                    level=1,
                )

            except Exception as exc:

                print(
                    "\nCould not isolate AAPL:"
                )

                print(
                    f"    {exc}"
                )

                return None

    price.index = pd.to_datetime(
        price.index
    )

    price = price.sort_index()

    price["close"] = pd.to_numeric(
        price["close"],
        errors="coerce",
    )

    price = price.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    price = price.dropna(
        subset=[
            "close"
        ]
    )

    print(
        "\nAAPL price data loaded."
    )

    print(
        f"\nRows:\n    {len(price):,}"
    )

    print(
        f"\nDate range:\n"
        f"    {price.index.min().date()}"
        f" -> "
        f"    {price.index.max().date()}"
    )

    print(
        "\nPrice statistics:"
    )

    print(
        price["close"].describe()
    )

    return price


# ============================================================
# PREPARE AAPL PLOT DATA
# ============================================================

def prepare_aapl_plot_data(
    aapl,
    price,
    all_results=None,
):

    print_header(
        "PREPARING AAPL PLOT DATA"
    )

    aapl = aapl.copy()
    price = price.copy()

    aapl.index = pd.to_datetime(aapl.index)
    price.index = pd.to_datetime(price.index)
    aapl = aapl.sort_index()
    price = price.sort_index()

    merged = aapl.join(price, how="left")

    merged["prediction_ma20"] = (
        merged["prediction"].rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    merged["actual_ma20"] = (
        merged["actual"].rolling(ROLLING_WINDOW, min_periods=5).mean()
    )

    # Time-series z-score is retained for visualization only.  It is not the
    # preferred way to turn an Alpha158 cross-sectional prediction into a
    # trading position.
    rolling_pred_mean = merged["prediction"].rolling(
        SIGNAL_LOOKBACK, min_periods=SIGNAL_MIN_PERIODS
    ).mean()
    rolling_pred_std = merged["prediction"].rolling(
        SIGNAL_LOOKBACK, min_periods=SIGNAL_MIN_PERIODS
    ).std()
    merged["prediction_z"] = (
        (merged["prediction"] - rolling_pred_mean)
        / (rolling_pred_std + EPS)
    ).replace([np.inf, -np.inf], np.nan)

    actual_mean = merged["actual"].mean()
    actual_std = merged["actual"].std()
    if pd.isna(actual_std) or actual_std < EPS:
        merged["actual_z"] = 0.0
    else:
        merged["actual_z"] = (merged["actual"] - actual_mean) / actual_std

    # --------------------------------------------------------
    # Cross-sectional rank signal
    # --------------------------------------------------------
    merged["prediction_rank"] = np.nan

    if all_results is not None and not all_results.empty:
        ranked = all_results.copy()
        if isinstance(ranked.index, pd.MultiIndex):
            names = list(ranked.index.names)
            date_level = "datetime" if "datetime" in names else names[-1]
            inst_level = "instrument" if "instrument" in names else names[0]
            tmp = ranked.reset_index()
            tmp[date_level] = pd.to_datetime(tmp[date_level])
            counts = tmp.groupby(date_level)["prediction"].transform("count")
            ranks = tmp.groupby(date_level)["prediction"].rank(pct=True)
            tmp["prediction_rank"] = ranks.where(counts >= PORTFOLIO_MIN_STOCKS)
            aapl_rank = tmp[tmp[inst_level].astype(str).str.upper() == "AAPL"]
            aapl_rank = aapl_rank.set_index(date_level)["prediction_rank"]
            merged["prediction_rank"] = merged.index.to_series().map(aapl_rank)

    # Fallback for unusual result indexes: use the AAPL time-series percentile.
    if merged["prediction_rank"].isna().all():
        merged["prediction_rank"] = merged["prediction"].expanding(
            min_periods=SIGNAL_MIN_PERIODS
        ).rank(pct=True)

    # Centered rank: +1 means strongest cross-sectional forecast and -1 means
    # weakest.  A dead zone around the cross-sectional median prevents weak
    # forecasts from becoming trades.
    centered_rank = 2.0 * merged["prediction_rank"] - 1.0
    raw_position = np.sign(centered_rank) * np.maximum(
        (np.abs(centered_rank) - SIGNAL_DEAD_ZONE)
        / (1.0 - SIGNAL_DEAD_ZONE),
        0.0,
    )
    merged["position"] = (
        pd.Series(raw_position, index=merged.index)
        .rolling(POSITION_SMOOTHING, min_periods=1)
        .mean()
        .clip(-POSITION_CLIP, POSITION_CLIP)
    )

    # --------------------------------------------------------
    # Label-aligned diagnostic return
    # --------------------------------------------------------
    # The Alpha158 label is close[t+2] / close[t+1] - 1.  The old version
    # shifted the position and multiplied it by close[t+1]/close[t]-1, which
    # evaluated a different return interval.  This diagnostic intentionally
    # aligns the model prediction with the exact return represented by its
    # label.  It is still a diagnostic, not a full execution backtest.
    merged["daily_return"] = (
        merged["close"].shift(-1) / merged["close"] - 1.0
    )
    merged["raw_alpha158_return"] = (
        merged["close"].shift(-2) / merged["close"].shift(-1) - 1.0
    )
    merged["buy_hold_return"] = merged["raw_alpha158_return"]
    merged["executed_position"] = merged["position"]

    merged["turnover"] = (
        merged["executed_position"].diff().abs().fillna(0.0)
    )
    transaction_cost = merged["turnover"] * (TRANSACTION_COST_BPS / 10000.0)
    merged["strategy_return_gross"] = (
        merged["executed_position"] * merged["raw_alpha158_return"]
    )
    merged["strategy_return"] = (
        merged["strategy_return_gross"] - transaction_cost
    )

    strategy_ret = merged["strategy_return"].fillna(0.0)
    gross_strategy_ret = merged["strategy_return_gross"].fillna(0.0)
    buy_hold_ret = merged["buy_hold_return"].fillna(0.0)

    merged["strategy_equity"] = 100.0 * (1.0 + strategy_ret).cumprod()
    merged["strategy_equity_gross"] = 100.0 * (1.0 + gross_strategy_ret).cumprod()
    merged["buy_hold_equity"] = 100.0 * (1.0 + buy_hold_ret).cumprod()

    valid_strategy = merged["strategy_return"].dropna()
    valid_gross = merged["strategy_return_gross"].dropna()
    valid_bh = merged["buy_hold_return"].dropna()

    def _annualized_sharpe(returns):
        if len(returns) < 2:
            return np.nan
        std = returns.std()
        if pd.isna(std) or std < EPS:
            return np.nan
        return returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)

    def _max_drawdown(equity):
        if equity.empty:
            return np.nan
        peak = equity.cummax()
        return (equity / peak - 1.0).min()

    strategy_days = len(valid_strategy)
    years = strategy_days / TRADING_DAYS_PER_YEAR
    if years > 0 and len(merged) > 1:
        strategy_cagr = (merged["strategy_equity"].iloc[-1] / 100.0) ** (1.0 / years) - 1.0
        bh_cagr = (merged["buy_hold_equity"].iloc[-1] / 100.0) ** (1.0 / years) - 1.0
    else:
        strategy_cagr = np.nan
        bh_cagr = np.nan

    strategy_metrics = {
        "observations": len(merged),
        "strategy_days": strategy_days,
        "signal_type": "AAPL cross-sectional prediction percentile",
        "label_aligned": True,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "gross_total_return": merged["strategy_equity_gross"].iloc[-1] / 100.0 - 1.0,
        "net_total_return": merged["strategy_equity"].iloc[-1] / 100.0 - 1.0,
        "buy_hold_total_return": merged["buy_hold_equity"].iloc[-1] / 100.0 - 1.0,
        "strategy_cagr": strategy_cagr,
        "buy_hold_cagr": bh_cagr,
        "strategy_sharpe": _annualized_sharpe(valid_strategy),
        "gross_strategy_sharpe": _annualized_sharpe(valid_gross),
        "buy_hold_sharpe": _annualized_sharpe(valid_bh),
        "strategy_max_drawdown": _max_drawdown(merged["strategy_equity"]),
        "buy_hold_max_drawdown": _max_drawdown(merged["buy_hold_equity"]),
        "average_abs_position": merged["executed_position"].abs().mean(),
        "average_turnover": merged["turnover"].mean(),
        "positive_return_fraction": (valid_strategy > 0).mean(),
        "aapl_prediction_actual_corr": merged["prediction"].corr(merged["actual"]),
        "aapl_prediction_raw_return_corr": merged["prediction"].corr(merged["raw_alpha158_return"]),
    }
    merged.attrs["strategy_metrics"] = strategy_metrics

    print("\nAAPL strategy metrics:")
    for key, value in strategy_metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    merged = merged.dropna(subset=["prediction", "actual"])

    print(f"""
Merged AAPL plot data:

    {merged.shape}
""")
    print("\nPrediction statistics:")
    print(merged["prediction"].describe())
    print("\nActual target statistics:")
    print(merged["actual"].describe())
    print("\nRaw Alpha158 return statistics:")
    print(merged["raw_alpha158_return"].describe())
    print("\nPrediction / actual correlation:")
    print(f"    {merged['prediction'].corr(merged['actual']):.6f}")

    return merged


# ============================================================
# CROSS-SECTIONAL MODEL DIAGNOSTIC
# ============================================================

def build_cross_sectional_strategy(results):
    """Evaluate the model in the cross-sectional way Alpha158 is designed for."""

    print_header("CROSS-SECTIONAL PORTFOLIO DIAGNOSTIC")

    if results is None or results.empty or not isinstance(results.index, pd.MultiIndex):
        print("\nCross-sectional diagnostic skipped: invalid results index.")
        return None

    names = list(results.index.names)
    if "instrument" not in names:
        print("\nCross-sectional diagnostic skipped: no instrument level.")
        return None
    date_level = "datetime" if "datetime" in names else next(
        (n for n in names if n != "instrument"), None
    )
    if date_level is None:
        print("\nCross-sectional diagnostic skipped: no date level.")
        return None

    df = results.reset_index().copy()
    df[date_level] = pd.to_datetime(df[date_level])
    df = df.dropna(subset=["prediction", "actual"])

    # Cross-sectional rank is the natural signal for a CS-normalized label.
    df["rank"] = df.groupby(date_level)["prediction"].rank(pct=True)
    counts = df.groupby(date_level)["prediction"].transform("count")
    df = df[counts >= PORTFOLIO_MIN_STOCKS].copy()
    if df.empty:
        print("\nCross-sectional diagnostic skipped: too few stocks per date.")
        return None

    df["raw_return"] = df.groupby("instrument", sort=False)["prediction"].transform(
        lambda _: np.nan
    )
    # The target itself corresponds to close[t+2] / close[t+1] - 1.  We need
    # the raw return rather than the normalized target, so retrieve close data
    # directly from Qlib for the test period.
    try:
        from qlib.data import D
        instruments = sorted(df["instrument"].astype(str).unique().tolist())
        close = D.features(
            instruments,
            ["$close"],
            start_time=TEST_START,
            end_time=TEST_END,
            freq="day",
        )
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [str(c[-1]) for c in close.columns]
        else:
            close.columns = [str(c) for c in close.columns]
        if "$close" not in close.columns:
            raise RuntimeError("Qlib close column not found.")
        close = close[["$close"]].rename(columns={"$close": "close"})
        if isinstance(close.index, pd.MultiIndex):
            close = close.reset_index()
            close[date_level] = pd.to_datetime(close[date_level])
            inst_col = "instrument" if "instrument" in close.columns else close.columns[0]
            close = close.rename(columns={inst_col: "instrument"})
        else:
            raise RuntimeError("Expected Qlib close data with instrument/date index.")
        close = close.sort_values(["instrument", date_level])
        close["raw_return"] = close.groupby("instrument")["close"].shift(-2) / close.groupby(
            "instrument"
        )["close"].shift(-1) - 1.0
        df = df.drop(columns=["raw_return"]).merge(
            close[["instrument", date_level, "raw_return"]],
            on=["instrument", date_level],
            how="left",
        )
    except Exception as exc:
        print(f"\nCross-sectional raw-return construction failed: {exc}")
        return None

    # Long top 20%, short bottom 20%, dollar neutral.  Equal-weight within
    # each side makes the diagnostic insensitive to stock price level.
    df["long_flag"] = df["rank"] >= 1.0 - PORTFOLIO_LONG_QUANTILE
    df["short_flag"] = df["rank"] <= PORTFOLIO_SHORT_QUANTILE
    df["long_weight"] = df["long_flag"].astype(float)
    df["short_weight"] = df["short_flag"].astype(float)
    long_n = df.groupby(date_level)["long_flag"].transform("sum").replace(0, np.nan)
    short_n = df.groupby(date_level)["short_flag"].transform("sum").replace(0, np.nan)
    df["position"] = df["long_weight"] / long_n - df["short_weight"] / short_n

    df["portfolio_return_gross"] = df["position"] * df["raw_return"]
    daily = df.groupby(date_level)["portfolio_return_gross"].sum(min_count=1)
    daily = daily.dropna().sort_index()

    # Turnover is one-half the absolute change in dollar-neutral weights.
    weights = df.pivot_table(index=date_level, columns="instrument", values="position", fill_value=0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover = turnover.reindex(daily.index).fillna(0.0)
    costs = turnover * (PORTFOLIO_TRANSACTION_COST_BPS / 10000.0)
    net = daily - costs

    gross_equity = 100.0 * (1.0 + daily).cumprod()
    net_equity = 100.0 * (1.0 + net).cumprod()

    # Daily cross-sectional IC is the most direct model-quality measure.
    ic = df.groupby(date_level).apply(
        lambda g: g["prediction"].corr(g["actual"])
    ).dropna()
    rank_ic = df.groupby(date_level).apply(
        lambda g: g["prediction"].corr(g["actual"], method="spearman")
    ).dropna()

    def sharpe(x):
        return np.nan if x.std() < EPS else x.mean() / x.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def max_dd(eq):
        return (eq / eq.cummax() - 1.0).min() if not eq.empty else np.nan

    years = len(net) / TRADING_DAYS_PER_YEAR
    metrics = {
        "dates": len(net),
        "stocks": df["instrument"].nunique(),
        "long_quantile": PORTFOLIO_LONG_QUANTILE,
        "short_quantile": PORTFOLIO_SHORT_QUANTILE,
        "transaction_cost_bps": PORTFOLIO_TRANSACTION_COST_BPS,
        "mean_ic": ic.mean(),
        "mean_rank_ic": rank_ic.mean(),
        "ic_positive_fraction": (ic > 0).mean(),
        "rank_ic_positive_fraction": (rank_ic > 0).mean(),
        "gross_total_return": gross_equity.iloc[-1] / 100.0 - 1.0,
        "net_total_return": net_equity.iloc[-1] / 100.0 - 1.0,
        "net_cagr": (net_equity.iloc[-1] / 100.0) ** (1.0 / years) - 1.0 if years > 0 else np.nan,
        "gross_sharpe": sharpe(daily),
        "net_sharpe": sharpe(net),
        "net_max_drawdown": max_dd(net_equity),
        "average_daily_turnover": turnover.mean(),
    }

    portfolio = pd.DataFrame({
        "gross_return": daily,
        "net_return": net,
        "turnover": turnover,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "ic": ic.reindex(daily.index),
        "rank_ic": rank_ic.reindex(daily.index),
    })
    portfolio.attrs["metrics"] = metrics

    print("\nCross-sectional portfolio metrics:")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    return portfolio


def save_cross_sectional_diagnostic(portfolio):
    if portfolio is None or portfolio.empty:
        return
    out = OUTPUT_DIR / "cross_sectional_strategy.csv"
    portfolio.to_csv(out)
    metrics = portfolio.attrs.get("metrics", {})
    if metrics:
        pd.DataFrame([metrics]).to_csv(
            OUTPUT_DIR / "cross_sectional_strategy_metrics.csv", index=False
        )
    print(f"\nCross-sectional strategy output saved:\n\n    {out}")


def create_cross_sectional_strategy_plot(portfolio):
    if portfolio is None or portfolio.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.plot(portfolio.index, portfolio["net_equity"], linewidth=1.6, label="Top/Bottom Quantile Strategy (Net)")
    ax.plot(portfolio.index, portfolio["gross_equity"], linewidth=1.0, alpha=0.7, label="Strategy (Gross)")
    ax.axhline(100.0, linewidth=0.8, alpha=0.5)
    ax.set_title("S&P 500 Alpha158 + LightGBM — Cross-Sectional Strategy Diagnostic", fontweight="bold")
    ax.set_ylabel("Equity (Start = 100)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = OUTPUT_DIR / "cross_sectional_strategy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nCross-sectional strategy plot saved:\n\n    {out}")


# ============================================================
# CREATE AAPL DIAGNOSTIC PLOT
# ============================================================

def create_aapl_plot(
    aapl,
    price,
    all_results=None,
):

    print_header(
        "CREATING AAPL DIAGNOSTIC PLOT"
    )

    if aapl is None:

        print(
            "\nNo AAPL prediction data."
        )

        return None

    if price is None:

        print(
            "\nNo AAPL price data."
        )

        return None

    if aapl.empty:

        print(
            "\nAAPL prediction DataFrame is empty."
        )

        return None

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "\nmatplotlib is not installed."
        )

        return None

    merged = prepare_aapl_plot_data(
        aapl,
        price,
        all_results,
    )

    if merged.empty:

        print(
            "\nNo merged AAPL data."
        )

        return None

    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(
            17,
            16,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                2.0,
                1.4,
                1.4,
                1.5,
            ]
        },
    )

    ax_price = axes[0]

    ax_signal = axes[1]

    ax_target = axes[2]

    ax_equity = axes[3]

    # ========================================================
    # PANEL 1 — STOCK PRICE
    # ========================================================

    ax_price.plot(
        merged.index,
        merged["close"],
        color="black",
        linewidth=1.3,
        label="AAPL Close",
        zorder=3,
    )

    ax_price.fill_between(
        merged.index,
        merged["close"].values,
        alpha=0.05,
        color="black",
    )

    ax_price.set_title(
        "AAPL Stock Price",
        fontsize=12,
        fontweight="bold",
    )

    ax_price.set_ylabel(
        "Price ($)"
    )

    ax_price.grid(
        alpha=0.25
    )

    ax_price.legend(
        loc="upper left"
    )

    # ========================================================
    # PANEL 2 — MODEL PREDICTION
    #
    # IMPORTANT:
    # This panel has its OWN y-axis.
    #
    # This is the key fix.
    # ========================================================

    ax_signal.plot(
        merged.index,
        merged["prediction"],
        color="#1565C0",
        linewidth=0.7,
        alpha=0.55,
        label="LightGBM Prediction",
    )

    ax_signal.plot(
        merged.index,
        merged["prediction_ma20"],
        color="#0D47A1",
        linewidth=2.0,
        label="20-Day Prediction Mean",
    )

    ax_signal.axhline(
        0.0,
        color="black",
        linewidth=0.9,
        alpha=0.6,
    )

    ax_signal.set_title(
        "LightGBM Prediction — Dedicated Scale",
        fontsize=12,
        fontweight="bold",
    )

    ax_signal.set_ylabel(
        "Model Output"
    )

    ax_signal.grid(
        alpha=0.25
    )

    ax_signal.legend(
        loc="upper left"
    )

    # ========================================================
    # PANEL 3 — PREDICTION VS TARGET
    #
    # TWO Y AXES
    #
    # Prediction ~= 0.02 std
    # Actual ~= 1.0 std
    #
    # ========================================================

    ax_target_right = (
        ax_target.twinx()
    )

    line1 = ax_target.plot(
        merged.index,
        merged["prediction_z"],
        color="#1565C0",
        linewidth=0.9,
        alpha=0.85,
        label="Prediction Z-Score",
    )

    line2 = ax_target_right.plot(
        merged.index,
        merged["actual"],
        color="#D32F2F",
        linewidth=0.7,
        alpha=0.30,
        label="Actual Alpha158 Target",
    )

    ax_target.plot(
        merged.index,
        merged["actual_ma20"],
        color="#B71C1C",
        linewidth=1.8,
        alpha=0.85,
        label="Actual 20-Day Mean",
    )

    ax_target.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )

    ax_target.set_title(
        "Prediction vs Actual Alpha158 Target",
        fontsize=12,
        fontweight="bold",
    )

    ax_target.set_ylabel(
        "Prediction Z-Score",
        color="#1565C0",
    )

    ax_target_right.set_ylabel(
        "Actual Alpha158 Target",
        color="#D32F2F",
    )

    ax_target.tick_params(
        axis="y",
        labelcolor="#1565C0",
    )

    ax_target_right.tick_params(
        axis="y",
        labelcolor="#D32F2F",
    )

    ax_target.grid(
        alpha=0.25
    )

    lines = (
        line1
        + line2
        + [
            ax_target.lines[-1]
        ]
    )

    labels = [
        line.get_label()
        for line in lines
    ]

    ax_target.legend(
        lines,
        labels,
        loc="upper left",
    )

    # ========================================================
    # PANEL 4 — STRATEGY
    # ========================================================

    ax_equity.plot(
        merged.index,
        merged["strategy_equity"],
        color="#1565C0",
        linewidth=1.5,
        label="AAPL Rank-Based Strategy",
    )

    ax_equity.plot(
        merged.index,
        merged["buy_hold_equity"],
        color="#333333",
        linewidth=1.2,
        alpha=0.75,
        label="AAPL Buy & Hold",
    )

    ax_equity.axhline(
        100.0,
        color="black",
        linewidth=0.8,
        alpha=0.4,
    )

    ax_equity.set_title(
        "Label-Aligned AAPL Signal Diagnostic",
        fontsize=12,
        fontweight="bold",
    )

    ax_equity.set_ylabel(
        "Equity (Start = 100)"
    )

    ax_equity.set_xlabel(
        "Date"
    )

    ax_equity.grid(
        alpha=0.25
    )

    ax_equity.legend(
        loc="upper left"
    )

    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        "AAPL — Alpha158 + LightGBM Diagnostics",
        fontsize=17,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        AAPL_PLOT_FILE,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        f"""
AAPL diagnostic plot saved:

    {AAPL_PLOT_FILE}
"""
    )

    return merged


# ============================================================
# CREATE DEDICATED SIGNAL PLOT
# ============================================================

def create_aapl_signal_plot(
    merged,
):

    print_header(
        "CREATING AAPL SIGNAL PLOT"
    )

    if merged is None:

        print(
            "\nNo merged AAPL data."
        )

        return

    if merged.empty:

        print(
            "\nMerged AAPL data is empty."
        )

        return

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            17,
            10,
        ),
        sharex=True,
    )

    # ========================================================
    # RAW MODEL OUTPUT
    # ========================================================

    axes[0].plot(
        merged.index,
        merged["prediction"],
        color="#1565C0",
        linewidth=0.7,
        alpha=0.45,
        label="Raw LightGBM Prediction",
    )

    axes[0].plot(
        merged.index,
        merged["prediction_ma20"],
        color="#0D47A1",
        linewidth=2.2,
        label="20-Day Rolling Prediction",
    )

    axes[0].axhline(
        0,
        color="black",
        linewidth=0.9,
    )

    axes[0].set_title(
        "AAPL Raw LightGBM Signal",
        fontweight="bold",
    )

    axes[0].set_ylabel(
        "Prediction"
    )

    axes[0].grid(
        alpha=0.25
    )

    axes[0].legend(
        loc="upper left"
    )

    # ========================================================
    # STANDARDIZED MODEL SIGNAL
    # ========================================================

    axes[1].plot(
        merged.index,
        merged["prediction_z"],
        color="#7B1FA2",
        linewidth=0.8,
        alpha=0.55,
        label="Prediction Z-Score",
    )

    axes[1].plot(
        merged.index,
        merged[
            "prediction_z"
        ]
        .rolling(
            ROLLING_WINDOW,
            min_periods=5,
        )
        .mean(),
        color="#4A148C",
        linewidth=2.0,
        label="20-Day Signal Mean",
    )

    axes[1].axhline(
        0,
        color="black",
        linewidth=0.9,
    )

    axes[1].set_title(
        "AAPL Standardized Model Signal",
        fontweight="bold",
    )

    axes[1].set_ylabel(
        "Z-Score"
    )

    axes[1].set_xlabel(
        "Date"
    )

    axes[1].grid(
        alpha=0.25
    )

    axes[1].legend(
        loc="upper left"
    )

    fig.suptitle(
        "AAPL — LightGBM Signal Detail",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        AAPL_SIGNAL_FILE,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        f"""
AAPL signal plot saved:

    {AAPL_SIGNAL_FILE}
"""
    )


# ============================================================
# CREATE STRATEGY PLOT
# ============================================================

def create_aapl_strategy_plot(
    merged,
):

    print_header(
        "CREATING AAPL STRATEGY PLOT"
    )

    if merged is None:

        return

    if merged.empty:

        return

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            17,
            10,
        ),
        sharex=True,
    )

    # ========================================================
    # POSITION
    # ========================================================

    ax1.plot(
        merged.index,
        merged["executed_position"],
        color="#1565C0",
        linewidth=0.8,
        alpha=0.65,
        label="Executed Position",
    )

    ax1.plot(
        merged.index,
        merged["executed_position"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=5,
        )
        .mean(),
        color="#0D47A1",
        linewidth=2.0,
        label="20-Day Executed Position Mean",
    )

    ax1.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax1.axhline(
        1,
        color="green",
        linewidth=0.5,
        alpha=0.4,
    )

    ax1.axhline(
        -1,
        color="red",
        linewidth=0.5,
        alpha=0.4,
    )

    ax1.set_ylim(
        -1.1,
        1.1,
    )

    ax1.set_title(
        "AAPL Cross-Sectional Rank Signal",
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Position"
    )

    ax1.grid(
        alpha=0.25
    )

    ax1.legend(
        loc="upper left"
    )

    # ========================================================
    # EQUITY
    # ========================================================

    ax2.plot(
        merged.index,
        merged["strategy_equity"],
        color="#1565C0",
        linewidth=1.6,
        label="AAPL Rank-Based Strategy",
    )

    ax2.plot(
        merged.index,
        merged["strategy_equity_gross"],
        color="#64B5F6",
        linewidth=1.0,
        alpha=0.75,
        label="AAPL Rank-Based Strategy (Gross)",
    )

    ax2.plot(
        merged.index,
        merged["buy_hold_equity"],
        color="black",
        linewidth=1.3,
        alpha=0.75,
        label="AAPL Buy & Hold",
    )

    ax2.axhline(
        100,
        color="gray",
        linewidth=0.8,
        alpha=0.6,
    )

    ax2.set_title(
        "AAPL Strategy Diagnostic — Net of 5 bps Turnover Cost",
        fontweight="bold",
    )

    ax2.set_ylabel(
        "Equity"
    )

    ax2.set_xlabel(
        "Date"
    )

    ax2.grid(
        alpha=0.25
    )

    ax2.legend(
        loc="upper left"
    )

    fig.suptitle(
        "AAPL — LightGBM Signal Strategy Diagnostic",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        AAPL_STRATEGY_FILE,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        f"""
AAPL strategy plot saved:

    {AAPL_STRATEGY_FILE}
"""
    )


# ============================================================
# SAVE AAPL STRATEGY METRICS
# ============================================================

def save_aapl_strategy_metrics(
    merged,
):

    if merged is None:
        return

    metrics = merged.attrs.get(
        "strategy_metrics",
        {},
    )

    if not metrics:
        return

    metrics_df = pd.DataFrame(
        [metrics]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        AAPL_STRATEGY_METRICS_FILE,
        index=False,
    )

    print(
        f"\nAAPL strategy metrics saved:\n\n    {AAPL_STRATEGY_METRICS_FILE}\n"
    )


# ============================================================
# LOG EXPERIMENT
# ============================================================

def run_with_qlib_experiment(
    model,
    dataset,
    X_train,
    y_train,
    X_valid,
    y_valid,
    results,
    importance_df,
):

    print_header(
        "LOGGING QLIB EXPERIMENT"
    )

    from qlib.workflow import R

    mlflow_uri = (
        f"sqlite:///{MLFLOW_DB.resolve()}"
    )

    print(
        f"""
Experiment:

    Alpha158_LightGBM

MLflow:

    {mlflow_uri}
"""
    )

    try:

        R.set_uri(
            mlflow_uri
        )

    except Exception as exc:

        print(
            "\nCould not explicitly set R URI:"
        )

        print(
            f"    {exc}"
        )

    try:

        with R.start(
            experiment_name="Alpha158_LightGBM"
        ):

            # =================================================
            # PARAMETERS
            # =================================================

            try:

                R.log_params(
                    model="LightGBM",
                    features="Alpha158",
                    train_start=TRAIN_START,
                    train_end=TRAIN_END,
                    valid_start=VALID_START,
                    valid_end=VALID_END,
                    test_start=TEST_START,
                    test_end=TEST_END,
                    n_features=len(
                        X_train.columns
                    ),
                    train_rows=len(
                        X_train
                    ),
                    valid_rows=len(
                        X_valid
                    ),
                    learning_rate=LEARNING_RATE,
                    num_leaves=NUM_LEAVES,
                    n_estimators=N_ESTIMATORS,
                    min_child_samples=MIN_CHILD_SAMPLES,
                    random_state=RANDOM_STATE,
                )

                print(
                    "\nParameters logged successfully."
                )

            except Exception as exc:

                print(
                    "\nCould not log parameters:"
                )

                print(
                    f"    {exc}"
                )

            # =================================================
            # METRICS
            # =================================================

            try:

                train_pred = model.predict(
                    X_train
                )

                valid_pred = model.predict(
                    X_valid
                )

                train_corr = (
                    pd.Series(
                        train_pred,
                        index=y_train.index,
                    )
                    .corr(
                        y_train
                    )
                )

                valid_corr = (
                    pd.Series(
                        valid_pred,
                        index=y_valid.index,
                    )
                    .corr(
                        y_valid
                    )
                )

                test_corr = (
                    results[
                        "prediction"
                    ]
                    .corr(
                        results["actual"]
                    )
                )

                R.log_metrics(
                    train_correlation=float(
                        train_corr
                    )
                    if pd.notna(
                        train_corr
                    )
                    else 0.0,

                    valid_correlation=float(
                        valid_corr
                    )
                    if pd.notna(
                        valid_corr
                    )
                    else 0.0,

                    test_correlation=float(
                        test_corr
                    )
                    if pd.notna(
                        test_corr
                    )
                    else 0.0,
                )

                print(
                    "\nLogged correlations:"
                )

                print(
                    f"    train: {train_corr:.6f}"
                )

                print(
                    f"    valid: {valid_corr:.6f}"
                )

                print(
                    f"    test:  {test_corr:.6f}"
                )

            except Exception as exc:

                print(
                    "\nCould not calculate/log metrics:"
                )

                print(
                    f"    {exc}"
                )

            # =================================================
            # MODEL
            # =================================================

            try:

                R.save_objects(
                    model=model
                )

                print(
                    "\nModel saved to Qlib recorder."
                )

            except Exception as exc:

                print(
                    "\nCould not save model artifact:"
                )

                print(
                    f"    {exc}"
                )

        print(
            "\nQlib experiment completed."
        )

    except Exception as exc:

        print(
            "\nWARNING:"
        )

        print(
            "Qlib experiment logging failed."
        )

        print(
            f"\nReason:\n    {exc}"
        )

        print(
            """
The model itself was already trained.

This logging error does NOT invalidate
the model or predictions.
"""
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QLIB ALPHA158 + LIGHTGBM"
    )
    print("=" * 70)

    print(
        f"""
Project root:

    {PROJECT_ROOT}

Qlib data:

    {QLIB_DATA_DIR}

Instrument file:

    {INSTRUMENT_FILE}

MLflow database:

    {MLFLOW_DB}
"""
    )

    # ========================================================
    # 1. CHECK ENVIRONMENT
    # ========================================================

    check_environment()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 2. LOAD INSTRUMENTS
    # ========================================================

    instruments = (
        load_instruments()
    )

    # ========================================================
    # 3. INITIALIZE QLIB
    # ========================================================

    initialize_qlib()

    # ========================================================
    # 4. CREATE ALPHA158
    # ========================================================

    handler = (
        create_alpha158(
            instruments
        )
    )

    # ========================================================
    # 5. CREATE DATASET
    # ========================================================

    dataset = (
        create_dataset(
            handler
        )
    )

    # ========================================================
    # 6. DATA KEY
    # ========================================================

    print_header(
        "SETTING DATA HANDLER KEY"
    )

    from qlib.data.dataset.handler import (
        DataHandlerLP
    )

    DATA_KEY = (
        DataHandlerLP.DK_L
    )

    print(
        f"""
DataHandlerLP.DK_L:

    {DATA_KEY}
"""
    )

    # ========================================================
    # 7. PREPARE DATA
    # ========================================================

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = prepare_data(
        dataset,
        DATA_KEY,
    )

    # ========================================================
    # 8. INSPECT FEATURES
    # ========================================================

    inspect_alpha158_features(
        X_train
    )

    # ========================================================
    # 9. CREATE MODEL
    # ========================================================

    model = (
        create_lightgbm()
    )

    # ========================================================
    # 10. TRAIN
    # ========================================================

    model = train_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    # ========================================================
    # 11. PREDICT
    # ========================================================

    prediction = predict(
        model,
        X_test,
    )

    # ========================================================
    # 12. RESULTS
    # ========================================================

    (
        results,
        correlation,
    ) = build_results(
        prediction,
        y_test,
    )

    # ========================================================
    # 13. PRIMARY CROSS-SECTIONAL DIAGNOSTIC
    # ========================================================

    cross_sectional = build_cross_sectional_strategy(
        results
    )

    save_cross_sectional_diagnostic(
        cross_sectional
    )

    create_cross_sectional_strategy_plot(
        cross_sectional
    )

    # ========================================================
    # 14. AAPL PREDICTIONS
    # ========================================================

    aapl = check_aapl(
        results
    )

    # ========================================================
    # 14. FEATURE IMPORTANCE
    # ========================================================

    importance_df = (
        feature_importance(
            model,
            X_train.columns.tolist(),
        )
    )

    # ========================================================
    # 15. SAVE CSV OUTPUTS
    # ========================================================

    save_outputs(
        results,
        importance_df,
    )

    # ========================================================
    # 16. LOAD AAPL PRICE
    # ========================================================

    aapl_price = (
        load_aapl_price()
    )

    # ========================================================
    # 17. CREATE AAPL PLOTS
    # ========================================================

    aapl_plot_data = (
        create_aapl_plot(
            aapl,
            aapl_price,
            results,
        )
    )

    # ========================================================
    # 18. CREATE DETAILED SIGNAL PLOT
    # ========================================================

    create_aapl_signal_plot(
        aapl_plot_data
    )

    # ========================================================
    # 19. CREATE STRATEGY PLOT
    # ========================================================

    create_aapl_strategy_plot(
        aapl_plot_data
    )

    save_aapl_strategy_metrics(
        aapl_plot_data
    )

    # ========================================================
    # 20. QLIB / MLFLOW
    # ========================================================

    run_with_qlib_experiment(
        model,
        dataset,
        X_train,
        y_train,
        X_valid,
        y_valid,
        results,
        importance_df,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "FINAL SUMMARY"
    )

    print(
        f"""
QLib data:

    {QLIB_DATA_DIR}

Instrument file:

    {INSTRUMENT_FILE}

Instruments:

    {len(instruments):,}

Alpha158 features:

    {len(X_train.columns):,}

Training observations:

    {len(X_train):,}

Validation observations:

    {len(X_valid):,}

Test observations:

    {len(X_test):,}

Best LightGBM iteration:

    {getattr(
        model,
        "best_iteration_",
        "N/A"
    )}

Test correlation:

    {correlation:.6f}

Predictions:

    {PREDICTIONS_FILE}

Feature importance:

    {FEATURE_IMPORTANCE_FILE}

AAPL diagnostic plot:

    {AAPL_PLOT_FILE}

AAPL signal plot:

    {AAPL_SIGNAL_FILE}

AAPL strategy plot:

    {AAPL_STRATEGY_FILE}

AAPL strategy metrics:

    {AAPL_STRATEGY_METRICS_FILE}

MLflow database:

    {MLFLOW_DB}
"""
    )

    print(
        "=" * 70
    )

    print(
        "\nDONE."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
