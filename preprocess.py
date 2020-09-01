import pandas as pd

df = pd.read_csv(r"data/Iris_raw.csv")
df.to_csv(r"data/Iris_preprocessed.csv", index = False)
