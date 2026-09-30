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
    chunk_size — зависит от того, как алгоритм дойдёт до конца.

    Параметры:
        df: DataFrame.
        chunk_size: желаемый минимальный размер чанка.
        column: имя колонки для группировки (по умолчанию "dt").
        assume_sorted: если True — предполагается, что df упорядочен по column.
            Если данные НЕ упорядочены — выбрасывается ValueError.
            Если False — данные сортируются внутри функции.

    Исключения:
        ValueError: если chunk_size <= 0, или если assume_sorted=True,
            но данные не упорядочены.
        KeyError: если column отсутствует в df.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if df.empty:
        return

    if assume_sorted:
        if not df[column].is_monotonic_increasing:
            raise ValueError(
                f"assume_sorted=True, но данные не упорядочены по '{column}'. "
                f"Передайте assume_sorted=False или отсортируйте данные."
            )
    else:
        df = df.sort_values(column).reset_index(drop=True)

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