"""Utility to *append* missing columns to an SQLiteModelStore table.

Unlike the full migration that streams rows through Python, this script:

1. Copies the original DB directory (so you keep a backup)
2. Opens the copied DB with the standard ``sqlite3`` module
3. Runs ``ALTER TABLE … ADD COLUMN …`` for every field defined on the
   provided ``SQLiteModelStore`` subclass that is *not already present* in
   the table.  Existing rows automatically receive ``NULL`` for each new
   column, preserving ``id`` and ``batch_time`` values.

Example (CLI):

```bash
python -m latentqa.datasets.alter_schema user_modeling.subject_inputs.SubjectInputDB \
$DATA_DIR/$HF_MODEL_NAME/synthsys/train_subject_input_db \
$DATA_DIR/$HF_MODEL_NAME/synthsys/train_subject_input_db_upd
```
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from importlib import import_module
from typing import Any, Type

from user_modeling.db_wrapper import SQLiteModelStore


def _ensure_copy(old_db_dir: str, new_db_dir: str, db_file_name: str) -> str:
    """Copy the DB file (and configs dir) from *old_db_dir* → *new_db_dir*.

    Returns the **path** to the copied DB file.
    """

    os.makedirs(new_db_dir, exist_ok=True)

    src_db_path = os.path.join(old_db_dir, db_file_name)
    dst_db_path = os.path.join(new_db_dir, db_file_name)

    if not os.path.exists(src_db_path):
        raise FileNotFoundError(f"No database file found at {src_db_path!s}")

    if os.path.exists(dst_db_path):
        raise FileExistsError(
            f"Destination already contains a database at {dst_db_path!s}; "
            "choose a new folder or delete the existing copy."
        )

    shutil.copy2(src_db_path, dst_db_path)

    # Copy configs folder when present (non-fatal if absent).
    src_cfg_dir = os.path.join(old_db_dir, "configs")
    if os.path.isdir(src_cfg_dir):
        dst_cfg_dir = os.path.join(new_db_dir, "configs")
        shutil.copytree(src_cfg_dir, dst_cfg_dir, dirs_exist_ok=True)

    return dst_db_path


def add_missing_columns(
    store_cls: Type[SQLiteModelStore],
    db_path: str,
    *,
    verbose: bool = True,
) -> None:
    """Issue ``ALTER TABLE`` statements for any *missing* columns.

    The set of desired columns (and their SQLite types) is taken from the
    subclass' ``_all_field_names`` / ``_all_field_types`` properties.
    """

    # We need an instance to access dynamic properties like table_name, keys, etc.
    store_instance = store_cls(db_dir=os.path.dirname(db_path), read_only=True)  # type: ignore[arg-type]

    table_name = store_instance.table_name

    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name});")
        existing_cols = {row[1] for row in cur.fetchall()}  # row[1] is column name

        desired_cols = store_instance._all_field_names  # type: ignore[attr-defined]
        col_types: dict[str, str] = store_instance._all_field_types  # type: ignore[attr-defined]

        missing_cols = [c for c in desired_cols if c not in existing_cols]

        # -----------------------------------------------
        # 1. Add any missing columns
        # -----------------------------------------------
        for col in missing_cols:
            col_type = col_types.get(col, "TEXT")
            if verbose:
                print(f"Adding column '{col}' ({col_type}) …")
            # Populate existing rows with the *string* "null" instead of SQL NULL.
            # The DEFAULT expression is applied to every existing row when the
            # column is added, so all prior data receives the literal "null".
            conn.execute(
                f"ALTER TABLE {table_name} " f"ADD COLUMN {col} {col_type} DEFAULT 'null';"
            )

        # -----------------------------------------------
        # 2. Recreate UNIQUE index to reflect current keys
        # -----------------------------------------------
        index_name = f"{table_name}_uniq"
        key_cols = store_instance._key_field_names  # type: ignore[attr-defined]
        cols_sql = ", ".join(key_cols)
        if verbose:
            print(f"Updating UNIQUE index '{index_name}' on ({cols_sql}) …")
        conn.execute(f"DROP INDEX IF EXISTS {index_name};")
        conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({cols_sql});")

        conn.commit()
        if verbose:
            if missing_cols:
                print("Schema update complete and UNIQUE index recreated.")
            else:
                print("UNIQUE index recreated (no new columns were needed).")
    finally:
        conn.close()


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Add missing columns to an SQLiteModelStore.")
    parser.add_argument(
        "store_cls",
        help="Dotted path to the DB wrapper class (e.g. latentqa.train_data.LQATrainingDataDB)",
    )
    parser.add_argument("old_db_dir", help="Directory that contains the *existing* DB file.")
    parser.add_argument(
        "new_db_dir", help="Directory where the updated copy of the DB will be written."
    )
    args = parser.parse_args()

    # Dynamically import wrapper class.
    module_path, _, class_name = args.store_cls.rpartition(".")
    mod = import_module(module_path)
    cls: Any = getattr(mod, class_name)

    if not issubclass(cls, SQLiteModelStore):  # type: ignore[arg-type]
        raise TypeError("Provided class is not a subclass of SQLiteModelStore.")

    # Instantiate once to discover the actual DB filename & table name.
    temp_instance = cls(db_dir=args.old_db_dir, read_only=True)  # type: ignore[arg-type]
    db_filename = os.path.basename(temp_instance.db_path)

    # Copy DB (& configs) to new location.
    copied_path = _ensure_copy(args.old_db_dir, args.new_db_dir, db_filename)

    # Add missing columns with ALTER TABLE.
    add_missing_columns(cls, copied_path)


if __name__ == "__main__":  # pragma: no cover
    main()
