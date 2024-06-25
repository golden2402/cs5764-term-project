import os
import json

from math import floor, log10
from typing import NamedTuple

import pandas as pd

from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash
from dash import dcc, html, callback, Input, Output


data_path = os.path.join("..", "app", "data")

# data loading & cleaning:
df_youtube = pd.read_csv(os.path.join(data_path, "US_youtube_trending_data.csv"))

df_youtube["category_name"] = df_youtube["category_id"].replace(
    {
        int(item["id"]): item["snippet"]["title"]
        for item in json.load(
            open(os.path.join(data_path, "US_category_id.json"), "r")
        )["items"]
    }
)

for column in ("published_at", "trending_date"):
    df_youtube[column] = pd.to_datetime(df_youtube[column]).dt.tz_localize(None)


# special trim for dislikes--after YouTube's removal of dislikes
df_youtube_dislikes = df_youtube[df_youtube["dislikes"] != 0]


def get_top_n_pairs(df: pd.DataFrame, feature: str):
    groups = df.groupby("category_name")
    groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

    return sorted(groups_pairs, key=lambda pair: pair[1])


NUMERIC_FEATURES = {
    "likes": "Likes",
    "dislikes": "Dislikes",
    "view_count": "Views",
    "comment_count": "Comments",
}

NUMERIC_TOP_PAIRS = {
    feature: get_top_n_pairs(df_youtube, feature) for feature in NUMERIC_FEATURES
}
NUMERIC_TOP_PAIRS["dislikes"] = get_top_n_pairs(df_youtube_dislikes, "dislikes")


# Dash:
def abbrev_num(value: int | float):
    n_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value)) / 3)
    value = round(value / 1000**n_thousands, 2)

    return f"{value:g}" + " KMBT"[n_thousands]


class Tab(NamedTuple):
    label: str
    value: str


TABS = (
    Tab("Chronology", "t1"),
    Tab("Categories (Bar)", "t2"),
    Tab("Categories (Pie)", "t3"),
    Tab("Downloads", "t4"),
)


app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div(
    (
        html.Header((html.H1("COVID YouTube Analytics"),)),
        dcc.Tabs(
            [dcc.Tab(label=label, value=value) for label, value in TABS],
            id="tabs",
            value=TABS[0].value,
        ),
        html.Div(id="tab-content"),
    )
)

server = app.server


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def update_tab(value: str | None):
    if value == TABS[0].value:
        return (html.H2("Chronology"), dcc.Loading((html.Div("t1-graph-container"))))
    elif value == TABS[1].value:
        return (
            html.Section((html.H2("Categories (Bar)"), html.P(""))),
            dcc.Loading((html.Div(id="t2-graph-container")), type="default"),
            html.Section(
                (
                    html.H4("Feature Selection:"),
                    dcc.Dropdown(
                        id="t2-feature-dropdown",
                        options=NUMERIC_FEATURES,
                        value=next(iter(NUMERIC_FEATURES)),
                    ),
                )
            ),
            html.Section(
                (
                    html.H4("Top Category Range:"),
                    dcc.RangeSlider(
                        id="t2-top-slider",
                        min=1,
                        max=15,
                        step=1,
                        value=[1, 15],
                    ),
                )
            ),
        )
    elif value == TABS[2].value:
        return (
            html.Section((html.H2("Categories (Pie)"), html.P(""))),
            dcc.Loading((html.Div(id="t3-graph-container")), type="default"),
            html.Section(
                (
                    html.H4("Feature Selection:"),
                    dcc.Dropdown(
                        id="t3-feature-dropdown",
                        options=NUMERIC_FEATURES,
                        value=next(iter(NUMERIC_FEATURES)),
                    ),
                )
            ),
            html.Section(
                (
                    html.H4("Top Category Range:"),
                    dcc.RangeSlider(
                        id="t3-top-slider",
                        min=1,
                        max=15,
                        step=1,
                        value=[1, 15],
                    ),
                )
            ),
        )
    # TODO: data section

    return None


# tab 2: Categories (Bar)
@callback(
    (
        Output("t2-graph-container", "children"),
        Output("t2-top-slider", "min"),
        Output("t2-top-slider", "max"),
    ),
    (Input("t2-feature-dropdown", "value"), Input("t2-top-slider", "value")),
)
def update_t2_bar(feature: str | None, range_min_max: list[int] | None):
    if feature is None:
        return html.P("Something went wrong--try picking a feature!")

    # data:
    n_min, n_max = range_min_max

    groups = NUMERIC_TOP_PAIRS[feature]
    groups_top_n = groups[
        (len(groups) + 1) - min(len(groups), n_max) - 1 : len(groups) - n_min + 1
    ]

    groups_zip = tuple(zip(*groups_top_n))

    # figure:
    feature_label = NUMERIC_FEATURES[feature]

    figure = go.Figure(
        go.Bar(
            x=groups_zip[0],
            y=groups_zip[1],
            text=tuple(map(abbrev_num, groups_zip[1])),
            hovertemplate="<b>%{x}</b>"
            + "<br>%{y:,} "
            + feature_label
            + "<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Categories: {feature_label} ({n_min} through {n_max})",
        xaxis_autorange="reversed",
        xaxis_title="Category",
        yaxis_title=feature_label,
    )

    return dcc.Graph(figure=figure), 1, len(groups)


# tab 2: Categories (Bar)
@callback(
    (
        Output("t3-graph-container", "children"),
        Output("t3-top-slider", "min"),
        Output("t3-top-slider", "max"),
    ),
    (Input("t3-feature-dropdown", "value"), Input("t3-top-slider", "value")),
)
def update_t3_bar(feature: str | None, range_min_max: list[int] | None):
    if feature is None:
        return html.P("Something went wrong--try picking a feature!")

    # data:
    n_min, n_max = range_min_max

    groups = NUMERIC_TOP_PAIRS[feature]
    groups_top_n = groups[
        (len(groups) + 1) - min(len(groups), n_max) - 1 : len(groups) - n_min + 1
    ]

    groups_zip = tuple(zip(*groups_top_n))

    # figure:
    feature_label = NUMERIC_FEATURES[feature]

    figure = go.Figure(
        go.Pie(
            labels=groups_zip[0],
            values=groups_zip[1],
            customdata=tuple(map(abbrev_num, groups_zip[1])),
            hovertemplate="<b>%{x}</b>"
            + "<br/>%{y:,} "
            + "<br/>%{percent}"
            + feature_label
            + "<extra></extra>",
            hole=0.3,
        )
    )
    figure.update_layout(
        title=f"Categories: {feature_label} ({n_min} through {n_max})",
    )

    return dcc.Graph(figure=figure), 1, len(groups)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
