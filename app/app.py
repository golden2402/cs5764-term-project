import os
import json

from math import floor, log10
from typing import NamedTuple
from datetime import date

import pandas as pd

from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash
from dash import dcc, html, callback, Input, Output, State


data_path = os.path.join("..", "app", "data")

# data loading & cleaning:
DF_CATEGORY_MAP = {
    int(item["id"]): item["snippet"]["title"]
    for item in json.load(open(os.path.join(data_path, "US_category_id.json"), "r"))[
        "items"
    ]
}

df_youtube = pd.read_csv(os.path.join(data_path, "US_youtube_trending_data.csv"))
df_youtube["category_name"] = df_youtube["category_id"].replace(DF_CATEGORY_MAP)

for column in ("published_at", "trending_date"):
    df_youtube[column] = pd.to_datetime(df_youtube[column]).dt.tz_localize(None)


# special trims:
# removal of all fields without dislikes--as per YouTube's removal of dislikes:
df_youtube_dislikes = df_youtube[df_youtube["dislikes"] != 0]

# COVID timeline:
COVID_MIN_DATE = date(2020, 3, 11)
COVID_MAX_DATE = date(2023, 5, 23)

df_youtube_covid_pre = df_youtube[
    (df_youtube["trending_date"].dt.date < COVID_MAX_DATE)
    & (df_youtube["trending_date"].dt.date > COVID_MIN_DATE)
]

df_youtube_covid_post = df_youtube[
    (df_youtube["trending_date"].dt.date >= COVID_MAX_DATE)
]

DOWNLOAD_MAP = {
    "full": df_youtube,
    "covid_pre": df_youtube_covid_pre,
    "covid_post": df_youtube_covid_post,
}

DOWNLOAD_MAP_LABELS = {
    "full": "Full Dataset",
    "covid_pre": "Pre-COVID (03/11/2020-05/23/2023)",
    "covid_post": "Post-COVID (05/23/2023-present)",
}


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
        return (
            html.Section((html.H2("Chronology"), html.P(""))),
            dcc.Loading((html.Div(id="t1-graph-container")), type="default"),
            html.Section(
                (
                    html.H3("Graph Refinements", className="label"),
                    html.Section(
                        (
                            html.H4("Category", className="label"),
                            dcc.Dropdown(
                                id="t1-filter-category",
                                options=DF_CATEGORY_MAP,
                                value=next(iter(DF_CATEGORY_MAP)),
                            ),
                        )
                    ),
                    html.Section(
                        (
                            html.H4("Filters", className="label"),
                            dcc.Checklist(
                                id="t1-filter-checklist",
                                options={"remove_dislikes": "Rows with Dislikes Only"},
                            ),
                        )
                    ),
                    html.Section(
                        (
                            html.H4("COVID Trimming", className="label"),
                            dcc.RadioItems(
                                id="t1-filter-radio",
                                options=DOWNLOAD_MAP_LABELS,
                                value=next(iter(DOWNLOAD_MAP_LABELS)),
                            ),
                        )
                    ),
                ),
            ),
        )
    elif value == TABS[1].value:
        return (
            html.Section((html.H2("Categories (Bar)"), html.P(""))),
            dcc.Loading((html.Div(id="t2-graph-container")), type="default"),
            html.Section(
                (
                    html.H4("Feature Selection:", className="label"),
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
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
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
                    html.H4("Feature Selection:", className="label"),
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
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                )
            ),
        )
    elif value == TABS[3].value:
        # may need to shift for extra tabs?
        return (
            html.Section((html.H2("Downloads"), html.P(""))),
            html.Section(
                (
                    html.H3("Cleaning Parameters", className="label"),
                    html.Div(
                        (
                            html.Section(
                                (
                                    html.H4("Dislike Filtering", className="label"),
                                    dcc.Checklist(
                                        id="t4-download-checklist",
                                        options={
                                            "remove_dislikes": "Rows with Dislikes Only"
                                        },
                                    ),
                                )
                            ),
                            html.Section(
                                (
                                    html.H4("COVID Trimming", className="label"),
                                    dcc.RadioItems(
                                        id="t4-download-radio",
                                        options=DOWNLOAD_MAP_LABELS,
                                        value=next(iter(DOWNLOAD_MAP_LABELS)),
                                    ),
                                )
                            ),
                        ),
                        className="download-parameters",
                    ),
                    html.Section(
                        (
                            html.H3("Finalize", className="label"),
                            html.Button("Clean & Download", id="t4-download-button"),
                        )
                    ),
                ),
            ),
            dcc.Store(id="t4-download-store"),
            dcc.Download(id="t4-download-output"),
        )

    return None


# tab 1: Chronology
@callback(
    Output("t1-graph-container", "children"),
    (
        Input("t1-filter-category", "value"),
        Input("t1-filter-checklist", "value"),
        Input("t1-filter-radio", "value"),
    ),
)
def update_t1_line(
    category_id: str | None, filters: list[str] | None, data_range: str | None
):
    df: pd.DataFrame = DOWNLOAD_MAP.get(data_range, next(iter(DOWNLOAD_MAP.values())))

    if category_id:
        df = df[df["category_id"] == int(category_id)]

    if len(filters or []) > 0:
        if "remove_dislikes" in filters:
            df = df[df["dislikes"] != 0]
        # TODO: other filters--does order matter?

    figure = go.Figure(layout=dict(showlegend=True, hovermode="x"))
    figure.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#d3d3d3",
        showline=True,
        showgrid=True,
    )

    for feature, feature_label in NUMERIC_FEATURES.items():
        df_isolated = df[df[feature] != 0]

        groups = df_isolated.groupby([df_isolated["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )
        groups_zip = tuple(zip(*groups_pairs))

        # bad results?:
        if len(groups_zip) == 0:
            continue

        figure.add_trace(
            go.Scatter(
                x=groups_zip[0],
                y=groups_zip[1],
                mode="lines",
                name=feature_label,
                hovertemplate="<b>%{x}</b>"
                + "<br>%{customdata} "
                + feature_label
                + "<extra></extra>",
                customdata=tuple(map(abbrev_num, groups_zip[1])),
            )
        )

    if len(figure.data) == 0:
        return html.Div(
            (
                html.H1(
                    "No data matches your particular filter set!",
                    style={"marginBottom": 0},
                ),
                html.P(
                    "Note: Dislikes were removed in November 2021, so they cannot show up post-COVID.",
                    style={"marginTop": 0},
                ),
            ),
            className="download-parameters",
        )

    return dcc.Graph(figure=figure)


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


# tab 3: Categories (Pie)
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
            hovertemplate="<b>%{label}</b>"
            + "<br>%{value:,} "
            + feature_label
            + "<br>%{percent}"
            + "<extra></extra>",
            hole=0.3,
        )
    )
    figure.update_layout(
        title=f"Categories: {feature_label} ({n_min} through {n_max})",
    )

    return dcc.Graph(figure=figure), 1, len(groups)


@callback(
    Output("t4-download-store", "data"),
    (Input("t4-download-checklist", "value"), Input("t4-download-radio", "value")),
)
def t4_set_params(value_checklist: list, value_radio: list):
    return {"options": value_checklist, "data_range": value_radio}


# tab 4: cleaning & download
@callback(
    Output("t4-download-output", "data"),
    Input("t4-download-button", "n_clicks"),
    State("t4-download-store", "data"),
    prevent_initial_call=True,
)
def t4_download(_: int, data: dict):
    options: list[str] = data["options"] or []
    df: pd.DataFrame = DOWNLOAD_MAP.get(
        data["data_range"], next(iter(DOWNLOAD_MAP.values()))
    )

    if "remove_dislikes" in options:
        df = df[df["dislikes"] != 0]

    return dcc.send_data_frame(df.to_csv, "download.csv")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
