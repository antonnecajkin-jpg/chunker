import pandas as pd
from chunker import split_into_chunks

dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
df = pd.DataFrame({"dt": dfs.repeat(3)})

print("Исходный df:")
print(df.head(10))
print(f"Всего строк: {len(df)}\n")

chunks = split_into_chunks(df, chunk_size=4)

print(f"Получено чанков: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"Чанк {i}: {len(chunk)} строк, "
          f"dt от {chunk['dt'].min()} до {chunk['dt'].max()}")