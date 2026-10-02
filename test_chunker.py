import pytest
import pandas as pd
import numpy as np
from chunker import split_into_chunks


@pytest.fixture
def standard_df():
    dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
    return pd.DataFrame({"dt": dfs.repeat(3)})


@pytest.fixture
def empty_df():
    return pd.DataFrame({"dt": pd.Series([], dtype="datetime64[ns]")})


@pytest.fixture
def unsorted_df():
    dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def check_all_requirements(chunks, chunk_size, column="dt"):
    chunks = list(chunks)

    total = sum(len(chunk) for chunk in chunks)
    assert total > 0

    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk) >= chunk_size

    assert len(chunks[-1]) > 0

    for i, chunk in enumerate(chunks):
        chunk_values = set(chunk[column].unique())
        for j, other in enumerate(chunks):
            if i == j:
                continue
            common = chunk_values & set(other[column].unique())
            assert not common

    for i in range(len(chunks) - 1):
        assert chunks[i][column].max() < chunks[i + 1][column].min()

    for i in range(len(chunks) - 1):
        assert chunks[i].index[-1] < chunks[i + 1].index[0]


def test_regular_case(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=4))
    assert len(chunks) == 3
    assert all(len(c) == 6 for c in chunks)
    check_all_requirements(chunks, chunk_size=4)


def test_chunk_size_bigger_than_df(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=100))
    assert len(chunks) == 1
    assert len(chunks[0]) == 18


def test_chunk_size_one(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=1))
    assert len(chunks) == 6
    assert all(len(c) == 3 for c in chunks)
    check_all_requirements(chunks, chunk_size=1)


def test_chunk_size_zero(standard_df):
    with pytest.raises(ValueError):
        list(split_into_chunks(standard_df, chunk_size=0))


def test_chunk_size_negative(standard_df):
    with pytest.raises(ValueError):
        list(split_into_chunks(standard_df, chunk_size=-5))


def test_empty_df(empty_df):
    chunks = list(split_into_chunks(empty_df, chunk_size=4))
    assert chunks == []


def test_unsorted_df(unsorted_df):
    chunks = list(split_into_chunks(unsorted_df, chunk_size=4, assume_sorted=False))
    assert len(chunks) == 3
    check_all_requirements(chunks, chunk_size=4)


def test_single_large_group():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01"] * 10)})
    chunks = list(split_into_chunks(df, chunk_size=4))
    assert len(chunks) == 1
    assert len(chunks[0]) == 10


def test_uneven_groups():
    df = pd.DataFrame({
        "dt": pd.to_datetime(
            ["2023-01-01"] * 1 +
            ["2023-01-02"] * 5 +
            ["2023-01-03"] * 1 +
            ["2023-01-04"] * 5
        )
    })
    chunks = list(split_into_chunks(df, chunk_size=4))
    check_all_requirements(chunks, chunk_size=4)
    assert sum(len(c) for c in chunks) == 12


def test_assume_sorted_true(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=4, assume_sorted=True))
    assert len(chunks) == 3
    check_all_requirements(chunks, chunk_size=4)


def test_assume_sorted_true_but_unsorted(unsorted_df):
    with pytest.raises(ValueError, match="assume_sorted=True"):
        list(split_into_chunks(unsorted_df, chunk_size=4, assume_sorted=True))


def test_single_group():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01"] * 7)})
    chunks = list(split_into_chunks(df, chunk_size=3))
    assert len(chunks) == 1
    assert len(chunks[0]) == 7


def test_all_data_preserved(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=4))
    combined = pd.concat(chunks)
    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True),
        standard_df.reset_index(drop=True),
    )


def test_groups_not_broken(standard_df):
    chunks = list(split_into_chunks(standard_df, chunk_size=4))
    for dt_value in standard_df["dt"].unique():
        chunks_with_dt = [
            i for i, chunk in enumerate(chunks)
            if (chunk["dt"] == dt_value).any()
        ]
        assert len(chunks_with_dt) == 1


def test_custom_column():
    df = pd.DataFrame({
        "event_type": ["A"] * 1 + ["B"] * 5 + ["C"] * 1 + ["D"] * 5
    })
    chunks = list(split_into_chunks(df, chunk_size=4, column="event_type"))
    check_all_requirements(chunks, chunk_size=4, column="event_type")
    assert sum(len(c) for c in chunks) == 12


def test_missing_column():
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01"] * 5)})
    with pytest.raises(KeyError):
        list(split_into_chunks(df, chunk_size=4, column="nonexistent"))


def test_empty_df_without_column():
    df = pd.DataFrame({"other": []})
    with pytest.raises(KeyError):
        list(split_into_chunks(df, chunk_size=4, column="dt"))


@pytest.mark.parametrize("chunk_size, expected_sizes", [
    (1, [2, 3, 1]),
    (2, [2, 3, 1]),
    (3, [5, 1]),
    (4, [5, 1]),
    (5, [5, 1]),
    (6, [6]),
])
def test_tz_example(chunk_size, expected_sizes):
    df = pd.DataFrame({
        "dt": pd.to_datetime([
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:03",
        ])
    })
    chunks = list(split_into_chunks(df, chunk_size=chunk_size))
    assert [len(c) for c in chunks] == expected_sizes




def test_sort_is_stable():
    """sort_values с kind='stable' сохраняет порядок одинаковых ключей."""
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2023-01-01"] * 3),
        "value": [10, 20, 30],
    })
    chunks = list(split_into_chunks(df, chunk_size=2))
    combined = pd.concat(chunks)
    assert list(combined["value"]) == [10, 20, 30]

def test_pd_na_raises():
    """pd.NA в колонке → ValueError, а не TypeError."""
    df = pd.DataFrame({"dt": ["a", "b", pd.NA, "c"]})
    with pytest.raises(ValueError, match="NA values"):
        list(split_into_chunks(df, chunk_size=2))


def test_nan_raises():
    """NaN в колонке → ValueError."""
    df = pd.DataFrame({"dt": [1.0, 2.0, np.nan, 4.0]})
    with pytest.raises(ValueError, match="NA values"):
        list(split_into_chunks(df, chunk_size=2))


def test_nat_raises():
    """NaT в datetime-колонке → ValueError."""
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2023-01-01", pd.NaT, "2023-01-02"])
    })
    with pytest.raises(ValueError, match="NA values"):
        list(split_into_chunks(df, chunk_size=2))