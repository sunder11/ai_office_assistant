import pandas as pd

# Sample DataFrame
data = {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
df = pd.DataFrame(data)

# Get 2 random rows and test and testing
sample_df = df.sample(n=2)
print(sample_df)
