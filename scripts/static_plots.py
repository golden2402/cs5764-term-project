import os

import pandas as pd


df = pd.read_csv(os.path.join("..", "app", "data", "US_youtube_trending_data.csv"))
df["trending_date"] = pd.to_datetime(
    df["trending_date"], format="%y.%d.%m", utc=True
).dt.tz_localize(None)
df["publish_time"] = pd.to_datetime(df["publish_time"]).dt.tz_localize(None)
