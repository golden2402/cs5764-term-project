import os
import json

import pandas as pd


data_path = os.path.join("..", "app", "data")

# cleaning:
df = pd.read_csv(os.path.join(data_path, "US_youtube_trending_data.csv"))

df["category_name"] = df["category_id"].replace(
    {
        int(item["id"]): item["snippet"]["title"]
        for item in json.load(
            open(os.path.join(data_path, "US_category_id.json"), "r")
        )["items"]
    }
)

for column in ("published_at", "trending_date"):
    df[column] = pd.to_datetime(df[column])
