"""PySpark engine: the same preprocessing as the Pandas engine, executed by Spark.

What Spark does here (all as distributed DataFrame operations):
  * read the CSV with an explicit schema (no slow type inference pass)
  * cleaning     : rule flags + one-pass aggregation of violation counts, filtering,
                   duplicate removal (groupBy on all columns)
  * features     : per-account WINDOW functions (row_number / sum / lag) for the
                   past-only history features
  * aggregation  : account-level groupBy (counts, sums, distinct counterparties)
  * statistics   : group-by fraud statistics by type and by day
  * output       : Parquet written by Spark itself

Spark runs in local mode (all CPU cores) on a laptop. The same code runs
unchanged on a cluster by changing the master URL (FRAUD_SPARK_MASTER) and
reading from distributed storage (HDFS / S3 paths instead of a local file).

pyspark is imported lazily, so this module can be imported without it; a clear
MissingDependencyError is raised only when Spark is actually requested.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from functools import reduce
from operator import or_
from pathlib import Path

import pandas as pd

from config import (
    BALANCE_COLUMNS,
    BASE_TIMESTAMP,
    NIGHT_END_HOUR,
    REQUIRED_COLUMNS,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
    TYPE_CODES,
    VALID_TRANSACTION_TYPES,
)
from src.preprocessing.cleaning import RULE_NAMES
from src.preprocessing.schema import ACCOUNT_COLUMNS, OUTPUT_COLUMNS, STATS_COLUMNS
from src.utils.errors import EmptyDatasetError, MissingDependencyError
from src.utils.io_utils import remove_path, save_table, table_paths
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns, require_dependency

logger = get_logger("spark")


# --------------------------------------------------------------------------
# Session management
# --------------------------------------------------------------------------
def _java_found() -> bool:
    return shutil.which("java") is not None or bool(os.environ.get("JAVA_HOME"))


def spark_available() -> bool:
    """True if pyspark is installed and a Java runtime can be found."""
    return importlib.util.find_spec("pyspark") is not None and _java_found()


def _spark_modules():
    require_dependency("pyspark", "pyspark", "the Spark engine")
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    return F, T, Window


def get_spark(
    app_name: str = "fraud-graph",
    master: str = SPARK_MASTER,
    driver_memory: str = SPARK_DRIVER_MEMORY,
    shuffle_partitions: int = SPARK_SHUFFLE_PARTITIONS,
):
    """Create (or reuse) a local SparkSession tuned for a laptop."""
    require_dependency("pyspark", "pyspark", "the Spark engine")
    if not _java_found():
        raise MissingDependencyError(
            "Java was not found, but PySpark needs a JDK. Install JDK 17 (works with PySpark 3.5 and 4.x), "
            "then restart the terminal. Check with: java -version"
        )
    # make Spark workers use the same Python as this program (important on Windows)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))  # default 200 is far too many for one laptop
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_transactions_spark(spark, path: Path | str, nrows: int | None = None):
    """Read the CSV with an explicit schema and add ``tx_id`` (position in file)."""
    F, T, _ = _spark_modules()
    path = Path(path)
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError(f"The file is empty: {path}") from exc
    require_columns(pd.DataFrame(columns=header), REQUIRED_COLUMNS, f"file {path.name}")

    numeric_types = {
        "step": T.IntegerType(),
        "isFraud": T.IntegerType(),
        "isFlaggedFraud": T.IntegerType(),
        "amount": T.DoubleType(),
        **{column: T.DoubleType() for column in BALANCE_COLUMNS},
    }
    # schema follows the file's own column order, so reordered CSVs still work
    schema = T.StructType([T.StructField(c, numeric_types.get(c, T.StringType()), True) for c in header])
    sdf = spark.read.csv(str(path), header=True, schema=schema).select(*REQUIRED_COLUMNS)
    sdf = sdf.withColumn("tx_id", F.monotonically_increasing_id())  # increases in file order
    if nrows:
        sdf = sdf.limit(int(nrows))
    return sdf.select("tx_id", *REQUIRED_COLUMNS)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _violation_columns() -> dict:
    """Spark expressions that are True when a row VIOLATES a rule (same rules as cleaning.py)."""
    F, _, _ = _spark_modules()
    c = F.col

    def nz(expr):  # NULL comparison results count as "not violated"
        return F.coalesce(expr, F.lit(False))

    negative_balance = reduce(or_, [nz(c(b) < 0) for b in BALANCE_COLUMNS])
    return {
        "null_required": reduce(or_, [c(column).isNull() for column in REQUIRED_COLUMNS]),
        "negative_amount": nz(c("amount") < 0),
        "invalid_step": nz(c("step") < 1),
        "unknown_type": ~nz(c("type").isin(*VALID_TRANSACTION_TYPES)),
        "invalid_label": ~nz(c("isFraud").isin(0, 1)),
        "invalid_flag_column": ~nz(c("isFlaggedFraud").isin(0, 1)),
        "negative_balance": negative_balance,
        "self_transfer": nz(c("nameOrig") == c("nameDest")),
        "blank_account": nz(F.trim(c("nameOrig")) == "") | nz(F.trim(c("nameDest")) == ""),
    }


def clean_spark(sdf):
    """Return (clean Spark DataFrame, partial cleaning report).

    The violation counts come from ONE aggregation pass over the data. Call
    finalize_cleaning_report() once the clean row count is known.
    """
    F, _, _ = _spark_modules()
    c = F.col
    violations = _violation_columns()

    flagged = sdf
    for name in RULE_NAMES:
        flagged = flagged.withColumn(f"_v_{name}", violations[name])
    any_invalid = reduce(or_, [c(f"_v_{name}") for name in RULE_NAMES])

    aggregations = [
        F.count(F.lit(1)).alias("n_input"),
        F.sum(F.coalesce(c("amount") == 0, F.lit(False)).cast("int")).alias("zero_amount_rows"),
        F.sum(any_invalid.cast("int")).alias("n_invalid_removed"),
        *[F.sum(c(f"_v_{name}").cast("int")).alias(name) for name in RULE_NAMES],
    ]
    row = flagged.agg(*aggregations).collect()[0].asDict()
    n_input = int(row["n_input"] or 0)
    if n_input == 0:
        raise EmptyDatasetError("The dataset is empty (0 rows).")

    valid = flagged.filter(~any_invalid).select("tx_id", *REQUIRED_COLUMNS)
    # duplicate removal: keep the first occurrence (smallest tx_id) of each identical row
    clean = (
        valid.groupBy(*REQUIRED_COLUMNS)
        .agg(F.min("tx_id").alias("tx_id"))
        .select("tx_id", *REQUIRED_COLUMNS)
    )
    report = {
        "n_input": n_input,
        "rule_violations": {name: int(row[name] or 0) for name in RULE_NAMES},
        "n_invalid_removed": int(row["n_invalid_removed"] or 0),
        "zero_amount_rows": int(row["zero_amount_rows"] or 0),
    }
    return clean, report


def finalize_cleaning_report(report: dict, n_clean: int) -> dict:
    report = dict(report)
    report["n_clean"] = int(n_clean)
    report["duplicates_removed"] = int(report["n_input"] - report["n_invalid_removed"] - n_clean)
    return report


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def add_features_spark(sdf):
    """Same features as src/features/transaction_features.py, using Spark window functions."""
    F, _, Window = _spark_modules()
    c = F.col
    base_epoch = int(pd.Timestamp(BASE_TIMESTAMP, tz="UTC").timestamp())

    type_code = None
    for tx_type, code in TYPE_CODES.items():
        condition = c("type") == tx_type
        type_code = F.when(condition, F.lit(code)) if type_code is None else type_code.when(condition, F.lit(code))
    type_code = type_code.otherwise(F.lit(-1)).cast("int")

    out = (
        sdf.withColumn("timestamp", (F.lit(base_epoch) + (c("step") - 1).cast("long") * 3600).cast("timestamp"))
        .withColumn("hour_of_day", ((c("step") - 1) % 24).cast("int"))
        .withColumn("day", F.floor((c("step") - 1) / 24).cast("int"))
        .withColumn("is_night", (c("hour_of_day") < NIGHT_END_HOUR).cast("int"))
        .withColumn("log_amount", F.log1p(c("amount")))
        .withColumn("type_code", type_code)
        .withColumn("dest_is_merchant", c("nameDest").startswith("M").cast("int"))
        .withColumn("balance_error_orig", c("newbalanceOrig") + c("amount") - c("oldbalanceOrg"))
        .withColumn("balance_error_dest", c("oldbalanceDest") + c("amount") - c("newbalanceDest"))
        .withColumn("orig_balance_emptied", ((c("oldbalanceOrg") > 0) & (c("newbalanceOrig") == 0)).cast("int"))
        .withColumn(
            "amount_to_oldbalance_ratio",
            F.when(c("oldbalanceOrg") > 0, c("amount") / c("oldbalanceOrg")).otherwise(F.lit(0.0)),
        )
    )
    for tx_type in VALID_TRANSACTION_TYPES:
        out = out.withColumn(f"type_{tx_type}", (c("type") == tx_type).cast("int"))

    # past-only history: windows ordered by (step, tx_id), frame ends one row BEFORE the current row
    for key, prefix, with_mean in (("nameOrig", "orig", True), ("nameDest", "dest", False)):
        ordered = Window.partitionBy(key).orderBy("step", "tx_id")
        before = ordered.rowsBetween(Window.unboundedPreceding, -1)
        out = out.withColumn(f"{prefix}_prev_tx_count", (F.row_number().over(ordered) - 1).cast("int"))
        out = out.withColumn(f"{prefix}_prev_amount_sum", F.coalesce(F.sum("amount").over(before), F.lit(0.0)))
        if with_mean:
            out = out.withColumn(
                f"{prefix}_prev_amount_mean",
                F.when(
                    c(f"{prefix}_prev_tx_count") > 0,
                    c(f"{prefix}_prev_amount_sum") / c(f"{prefix}_prev_tx_count"),
                ).otherwise(F.lit(0.0)),
            )
        out = out.withColumn(
            f"{prefix}_steps_since_last_tx",
            F.coalesce(c("step") - F.lag("step", 1).over(ordered), F.lit(-1)).cast("int"),
        )
    return out.select(*OUTPUT_COLUMNS)


# --------------------------------------------------------------------------
# Aggregations and statistics
# --------------------------------------------------------------------------
def account_aggregates_spark(sdf):
    """One row per account (see src/features/account_aggregates.py for the meaning)."""
    F, _, _ = _spark_modules()
    c = F.col

    sent = sdf.groupBy(c("nameOrig").alias("account")).agg(
        F.count(F.lit(1)).alias("out_tx_count"),
        F.sum("amount").alias("out_amount_sum"),
        F.countDistinct("nameDest").alias("out_unique_counterparties"),
        F.min("step").alias("out_first_step"),
        F.max("step").alias("out_last_step"),
        F.sum("isFraud").alias("label_fraud_sent"),
    )
    received = sdf.groupBy(c("nameDest").alias("account")).agg(
        F.count(F.lit(1)).alias("in_tx_count"),
        F.sum("amount").alias("in_amount_sum"),
        F.countDistinct("nameOrig").alias("in_unique_counterparties"),
        F.min("step").alias("in_first_step"),
        F.max("step").alias("in_last_step"),
        F.sum("isFraud").alias("label_fraud_received"),
    )
    edges = sdf.select("nameOrig", "nameDest").distinct()
    both_directions = edges.select(c("nameOrig").alias("account"), c("nameDest").alias("counterparty")).union(
        edges.select(c("nameDest").alias("account"), c("nameOrig").alias("counterparty"))
    ).distinct()
    neighbours = both_directions.groupBy("account").agg(F.count(F.lit(1)).alias("n_counterparties"))

    acc = (
        sent.join(received, on="account", how="full_outer")
        .join(neighbours, on="account", how="left")
        .withColumn("first_step", F.least(c("out_first_step"), c("in_first_step")))
        .withColumn("last_step", F.greatest(c("out_last_step"), c("in_last_step")))
        .fillna(
            0,
            subset=[
                "out_tx_count", "out_amount_sum", "out_unique_counterparties", "label_fraud_sent",
                "in_tx_count", "in_amount_sum", "in_unique_counterparties", "label_fraud_received",
                "n_counterparties",
            ],
        )
        .withColumn("total_tx_count", c("out_tx_count") + c("in_tx_count"))
        .withColumn("total_amount", c("out_amount_sum") + c("in_amount_sum"))
        .withColumn(
            "out_amount_avg",
            F.when(c("out_tx_count") > 0, c("out_amount_sum") / c("out_tx_count")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "in_amount_avg",
            F.when(c("in_tx_count") > 0, c("in_amount_sum") / c("in_tx_count")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "avg_tx_amount",
            F.when(c("total_tx_count") > 0, c("total_amount") / c("total_tx_count")).otherwise(F.lit(0.0)),
        )
        .withColumn("is_merchant", c("account").startswith("M").cast("int"))
    )
    return acc.select(*ACCOUNT_COLUMNS)


def fraud_stats_spark(sdf) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-by fraud statistics by type and by day (tiny results, collected to Pandas)."""
    F, _, _ = _spark_modules()
    c = F.col

    def stats(key: str) -> pd.DataFrame:
        grouped = sdf.groupBy(key).agg(
            F.count(F.lit(1)).alias("n_transactions"),
            F.sum("isFraud").alias("n_fraud"),
            F.sum("amount").alias("total_amount"),
            F.sum(c("amount") * c("isFraud")).alias("fraud_amount"),
        )
        result = grouped.toPandas().sort_values(key).reset_index(drop=True)
        result["fraud_rate_pct"] = (100.0 * result["n_fraud"] / result["n_transactions"]).round(4)
        return result[[key, *STATS_COLUMNS]]

    return stats("type"), stats("day")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_table_spark(sdf, stem: Path | str) -> Path:
    """Write a Spark DataFrame as Parquet using Spark; fall back to Pandas if that fails.

    On Windows, Spark's own file writer often fails unless Hadoop's winutils is
    installed. In that case the data is collected to Pandas and written with
    pyarrow (fine for a laptop-sized sample, but it needs enough RAM).
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet_path, csv_path = table_paths(stem)
    remove_path(parquet_path)
    remove_path(csv_path)
    try:
        sdf.write.mode("overwrite").parquet(str(parquet_path))
        return parquet_path
    except Exception as exc:  # Py4J / Hadoop errors have many types
        first_line = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
        logger.warning(
            "Spark could not write Parquet for '%s' (%s). This is common on Windows without Hadoop winutils. "
            "Falling back to collecting the data to Pandas.",
            stem.name,
            first_line,
        )
        return save_table(sdf.toPandas(), stem)


# --------------------------------------------------------------------------
# Whole-engine entry point
# --------------------------------------------------------------------------
def run_spark_engine(input_path: Path | str, nrows: int | None, tx_stem: Path, acc_stem: Path) -> dict:
    """Run read -> clean -> features -> aggregates -> statistics -> write with Spark."""
    spark = get_spark()  # raises a friendly MissingDependencyError if pyspark / Java is missing
    from pyspark import StorageLevel

    try:
        logger.info("Spark %s running with master=%s", spark.version, spark.sparkContext.master)
        sdf = read_transactions_spark(spark, input_path, nrows)
        clean_sdf, cleaning = clean_spark(sdf)

        # persist: the features feed several outputs, so compute them once
        features = add_features_spark(clean_sdf).persist(StorageLevel.MEMORY_AND_DISK)
        n_clean = features.count()
        if n_clean == 0:
            raise EmptyDatasetError("No valid rows remain after cleaning.")
        cleaning = finalize_cleaning_report(cleaning, n_clean)

        accounts = account_aggregates_spark(features).persist(StorageLevel.MEMORY_AND_DISK)
        n_accounts = accounts.count()
        by_type, by_day = fraud_stats_spark(features)

        tx_path = write_table_spark(features, tx_stem)
        acc_path = write_table_spark(accounts, acc_stem)
        accounts.unpersist()
        features.unpersist()
    finally:
        spark.stop()

    return {
        "cleaning": cleaning,
        "n_accounts": int(n_accounts),
        "by_type": by_type,
        "by_day": by_day,
        "transactions_path": tx_path,
        "accounts_path": acc_path,
    }