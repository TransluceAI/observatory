from __future__ import annotations

import json
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Generator, Generic, Type, TypeVar

import aiosqlite
from pydantic import BaseModel
from tqdm import tqdm
from user_modeling.user_attribute import UserAttribute

T = TypeVar("T", bound=BaseModel)


class SQLiteModelStore(Generic[T]):
    """A lightweight, generic, async SQLite wrapper for persisting Pydantic models.

    The table schema is automatically derived from the model's fields, with one extra
    ``batch_time`` column that records when a batch of rows is inserted.
    """

    def __init__(self, db_dir: str, model_cls: Type[T], read_only: bool = False):
        self.model_cls = model_cls
        self.table_name = model_cls.__name__.lower()
        self.db_dir = db_dir
        self.db_path = os.path.join(db_dir, f"{self.table_name}.db")

        # If the caller only intends to read, remember that (used when we open connections
        # and when we guard write-methods like `insert`/`insert_batch`).
        self.read_only = read_only

        # Create folder to save configs if it doesn't exist.
        self.config_dir = os.path.join(db_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)

        # Initialize the database schema synchronously if not read-only.
        if not read_only:
            self._sync_init_schema()

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    @property
    def _all_field_names(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def _key_field_names(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def _all_field_types(self) -> dict[str, str]:
        return {name: "TEXT" for name in self._all_field_names}

    async def _open_db(self) -> aiosqlite.Connection:
        """
        Open an async connection (read-only if requested).
        """
        if self.read_only:
            # `mode=ro` opens the DB read-only and therefore cannot create WAL files nor
            # mutate schema.  URI must be enabled for this to work.
            return await aiosqlite.connect(
                f"file:{self.db_path}?mode=ro", uri=True, detect_types=sqlite3.PARSE_DECLTYPES
            )

        return await aiosqlite.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def _sync_init_schema(self) -> None:
        """
        Synchronously create the table and indices if they don't exist.
        Called from __init__ for non-read-only instances.
        """
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            # Enable WAL so concurrent readers don't block the writer.
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass

            # Build schema: id (PK), batch_time, then one column per model field.
            field_columns_parts: list[str] = []
            for name in self._all_field_names:
                col_type = self._all_field_types[name]
                field_columns_parts.append(f"{name} {col_type}")

            field_columns = ", ".join(field_columns_parts)
            create_stmt = (
                f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "batch_time TEXT, "
                f"{field_columns});"
            )
            conn.execute(create_stmt)

            unique_cols = ", ".join(self._key_field_names)
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {self.table_name}_uniq "
                f"ON {self.table_name} ({unique_cols});"
            )
            conn.commit()
        finally:
            conn.close()

    @asynccontextmanager
    async def get_db(self):  # pragma: no cover
        conn = await self._open_db()
        try:
            yield conn
        finally:
            await conn.close()

    @contextmanager
    def get_sync_db(self):
        if self.read_only:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, detect_types=sqlite3.PARSE_DECLTYPES
            )
        else:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            yield conn
        finally:
            conn.close()

    def _stringify_field(self, value: Any) -> str:
        # Explicit handling for common ML tensor / array types.
        if isinstance(value, UserAttribute):
            return value.value
        elif isinstance(value, type) and issubclass(value, UserAttribute):
            return value.__name__
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (str, int, float, type(None))):
            return str(value) if value is not None else "null"
        elif isinstance(value, BaseModel):
            return value.model_dump_json()
        else:
            return json.dumps(value, default=str)

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> T:
        raise NotImplementedError("Subclasses must implement this method.")

    def _save_config(self, config: BaseModel | dict[str, Any], batch_time: str):
        config_path = os.path.join(self.config_dir, f"{batch_time}.json")
        with open(config_path, "w") as f:
            if isinstance(config, BaseModel):
                f.write(config.model_dump_json())
            else:
                json.dump(config, f)

    def _build_where_clause(
        self, conditions: dict[str, Any] | None = None
    ) -> tuple[str, list[Any]]:
        """
        Construct a ``WHERE`` clause (plus its bound parameters) from the given ``conditions`` mapping.

        - scalar → equality check (``col = ?``)
        - collection → ``IN`` clause (``col IN (?, ?, ...)``)
        """
        where_sql = ""
        params: list[Any] = []
        if conditions:
            clauses: list[str] = []
            for col, val in conditions.items():
                if isinstance(val, (list, tuple, set)):
                    if not val:
                        raise ValueError(f"Empty value list provided for column '{col}'.")
                    placeholders = ", ".join(["?"] * len(val))
                    clauses.append(f"{col} IN ({placeholders})")
                    params.extend(self._stringify_field(v) for v in val)
                else:
                    clauses.append(f"{col} = ?")
                    params.append(self._stringify_field(val))
            where_sql = " WHERE " + " AND ".join(clauses)
        return where_sql, params

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_time_str(self) -> str:
        return datetime.now().strftime("%m%d%Y_%H:%M:%S")

    def get_config_dict(self, time_str: str) -> dict[str, Any] | None:
        config_path = os.path.join(self.config_dir, f"{time_str}.json")
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config for {time_str}: {e}")
            return None

    def convert_row_to_obj(self, row: list[Any]) -> T:
        raw_fields = row[2:]  # Skip id and batch_time.
        obj_dict = dict(zip(self._all_field_names, raw_fields))
        return self._convert_serialized_obj_to_obj(obj_dict)

    async def return_match(self, obj: T, return_raw: bool = False) -> T | list[Any] | None:
        conditions = []
        params = []
        for field_name in self._key_field_names:
            conditions.append(f"{field_name} = ?")
            params.append(self._stringify_field(getattr(obj, field_name)))
        select_stmt = f"SELECT * FROM {self.table_name} WHERE " + " AND ".join(conditions)
        async with self.get_db() as db:
            cursor = await db.execute(
                select_stmt,
                tuple(params),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return list(row) if return_raw else self.convert_row_to_obj(list(row))

    async def check_if_exists(self, obj: T) -> bool:
        return await self.return_match(obj) is not None

    async def delete(self, row_id: int):
        if self.read_only:
            raise RuntimeError("Cannot delete from read-only database.")
        async with self.get_db() as db:
            await db.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (row_id,))
            await db.commit()

    async def delete_where(self, **conditions: Any):
        """
        Delete *all* rows that satisfy the supplied ``conditions``.

        Conditions follow the same syntax as ``fetch_all``/``count``:
        - scalar value  → equality check (``col = value``)
        - iterable      → ``IN`` clause (``col IN (...)``)

        Example
        -------
        >>> await store.delete_where(batch_time="08122024_12:34:56")

        Notes
        -----
        • For safety, at least one condition must be provided—this method
          refuses to wipe the entire table by accident.
        • The in-memory row-cache is invalidated after deletion.
        """
        if self.read_only:
            raise RuntimeError("Cannot delete from read-only database.")

        if not conditions:
            raise ValueError("Refusing to delete without conditions (would wipe the table).")

        where_sql, params = self._build_where_clause(conditions)
        delete_stmt = f"DELETE FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            await db.execute(delete_stmt, tuple(params))
            await db.commit()

    async def insert(self, obj: T, time_str: str):
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database.")

        field_names = self._all_field_names
        placeholders = ", ".join(["?"] * (len(field_names) + 1))  # batch_time + fields
        cols_sql = ", ".join(["batch_time"] + field_names)
        insert_stmt = (
            f"INSERT OR IGNORE INTO {self.table_name} ({cols_sql}) VALUES ({placeholders})"
        )

        values: list[Any] = [time_str]
        for name in field_names:
            val = getattr(obj, name)
            values.append(self._stringify_field(val))

        async with self.get_db() as db:
            await db.execute(insert_stmt, tuple(values))
            await db.commit()

    async def insert_batch(
        self,
        objs: list[T],
        batch_time: str | None = None,
        config: BaseModel | None = None,
        verbose: bool = False,
    ):
        """Insert a batch of model instances.  All rows share the same ``batch_time``.
        ``batch_time`` defaults to current UTC ISO timestamp.
        """
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database.")

        if not objs:
            return

        if batch_time is None:
            batch_time = self.get_time_str()

        if config is not None:
            self._save_config(config, batch_time)

        field_names = self._all_field_names
        placeholders = ", ".join(["?"] * (len(field_names) + 1))  # batch_time + fields
        cols_sql = ", ".join(["batch_time"] + field_names)
        insert_stmt = (
            f"INSERT OR IGNORE INTO {self.table_name} ({cols_sql}) VALUES ({placeholders})"
        )

        rows: list[tuple[Any, ...]] = []
        if verbose:
            objs_iter = tqdm(objs, desc=f"Inserting objects into {self.table_name}")
        else:
            objs_iter = objs

        for obj in objs_iter:
            values: list[Any] = [batch_time]
            for name in field_names:
                val = getattr(obj, name)
                values.append(self._stringify_field(val))
            rows.append(tuple(values))

        async with self.get_db() as db:
            await db.executemany(insert_stmt, rows)
            await db.commit()

        if verbose:
            print(f"Inserted {len(rows)} rows into {self.table_name}")

    async def fetch_all(self, **conditions: Any) -> list[T]:
        """
        Return *all* rows satisfying ``conditions`` as Pydantic objects
        (ignores ``batch_time``/``id`` when constructing the objects).
        """
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()

        return [self.convert_row_to_obj(list(row)) for row in rows]

    async def fetch_all_raw(self, **conditions: Any) -> list[list[Any]]:
        """
        Return *all* rows satisfying ``conditions`` as Pydantic objects
        (ignores ``batch_time``/``id`` when constructing the objects).
        """
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()

        return [list(row) for row in rows]

    def sync_iterrows(self, **conditions: Any) -> Generator[T, None]:
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"
        with self.get_sync_db() as db:
            cursor = db.execute(query, tuple(params))
            for row in cursor:
                yield self.convert_row_to_obj(list(row))

    def sync_iterrows_raw(self, **conditions: Any) -> Generator[list[Any], None]:
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"
        with self.get_sync_db() as db:
            cursor = db.execute(query, tuple(params))
            for row in cursor:
                yield list(row)

    async def iterrows(self, **conditions: Any) -> AsyncGenerator[T, None]:
        """
        Asynchronously iterate over rows that satisfy ``conditions``, yielding
        Pydantic objects one-by-one.
        """
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            cursor = await db.execute(query, tuple(params))
            async for row in cursor:
                yield self.convert_row_to_obj(list(row))

    async def iterrows_raw(self, **conditions: Any) -> AsyncGenerator[list[Any], None]:
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT * FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            cursor = await db.execute(query, tuple(params))
            async for row in cursor:
                yield list(row)

    async def get_unique_values(self, column_names: list[str], **conditions: Any) -> list[Any]:
        where_sql, params = self._build_where_clause(conditions)
        async with self.get_db() as db:
            cursor = await db.execute(
                f"SELECT DISTINCT {', '.join(column_names)} FROM {self.table_name}{where_sql}",
                tuple(params),
            )
            rows = await cursor.fetchall()
            return [tuple(row) for row in rows]

    def __len__(self) -> int:
        with self.get_sync_db() as db:
            cursor = db.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cursor.fetchone()[0]

    def count(self, **conditions: Any) -> int:
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT COUNT(*) FROM {self.table_name}{where_sql}"
        with self.get_sync_db() as db:
            cursor = db.execute(query, tuple(params))
            return cursor.fetchone()[0]
