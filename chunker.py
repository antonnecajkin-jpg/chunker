import numpy as np
import pandas as pd
from typing import Iterator


def split_into_chunks(
    df: pd.DataFrame,
    chunk_size: int,
    column: str = "dt",
    assume_sorted: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Разбивает DataFrame на чанки по указанной колонке.

    Возвращает генератор чанков. Хвостовой чанк может быть меньше
    chunk_size.

    Параметры:
        df: DataFrame.
        chunk_size: желаемый минимальный размер чанка.
        column: имя колонки для группировки. По умолчанию "dt".
        assume_sorted: если True, данные считаются упорядоченными по column.
            Если нет, выбрасывается ValueError. По умолчанию False,
            данные сортируются внутри.

    Исключения:
        ValueError: если chunk_size <= 0.
        ValueError: если assume_sorted=True, но данные не упорядочены.
        ValueError: если column содержит NA (pd.NA, NaN, NaT).
        ValueError: если column содержит смешанные типы.
        KeyError: если column отсутствует в df.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if df.empty:
        return
    if df[column].isna().any():
        raise ValueError(
            f"Column '{column}' contains NA values. "
            f"Use fillna() or dropna() before calling."
    )
    if df[column].dtype == object:
        types = df[column].apply(type).unique()
        if len(types) > 1:
            raise ValueError(
                f"Column '{column}' contains mixed types. "
                f"Unify types before calling."
        )

    if assume_sorted:
        if not df[column].is_monotonic_increasing:
            raise ValueError(
                f"assume_sorted=True, но данные не упорядочены по '{column}'. "
                f"Передайте assume_sorted=False или отсортируйте данные."
            )
    else:
        df = df.sort_values(column, kind="stable").reset_index(drop=True)

    values = df[column].to_numpy()
    is_new_group = np.empty(len(values), dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = values[1:] != values[:-1]
    group_starts = np.flatnonzero(is_new_group)
    group_ends = np.append(group_starts[1:], len(values))
    group_sizes = group_ends - group_starts

    current_start = 0
    current_size = 0
    for start, size in zip(group_starts, group_sizes):
        current_size += size
        if current_size >= chunk_size:
            yield df.iloc[current_start:start + size]
            current_start = start + size
            current_size = 0

    if current_size > 0:
        yield df.iloc[current_start:len(values)]