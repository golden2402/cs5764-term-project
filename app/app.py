import os

import pandas as pd

import plotly.express as px
from dash import Dash, html


# df = pd.read_csv(os.path.join(".", "data", "US_youtube_trending_data.csv"))
# df["trending_date"] = pd.to_datetime(
#     df["trending_date"], format="%y.%d.%m", utc=True
# ).dt.tz_localize(None)
# df["publish_time"] = pd.to_datetime(df["publish_time"]).dt.tz_localize(None)


app = Dash(__name__)
app.layout = html.Div(
    (
        html.H1("Dash is running!"),
        html.P("Content goes here."),
    )
)

server = app.server


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
