"""Simple scratch script to test pandas column downcasting."""

import pandas as pd

df = pd.DataFrame({"a": [1.0, 2.0], "b": [1, 2]})
float_cols = df.select_dtypes(include=["float64"]).columns
# float_cols is now populated
