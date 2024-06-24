import os
import json

from typing import NamedTuple
from math import floor, log10
from datetime import date, datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)
np.set_printoptions(precision=2)


data_path = os.path.join("..", "app", "data")

# cleaning:
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


def abbrev_num(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value)) / 3)
    value = round(value / 1000**num_thousands, 2)

    return f"{value:g}" + " KMBT"[num_thousands]


class SubplotFeature(NamedTuple):
    id: str
    display_name: str


covid_min_date = date(2020, 8, 11)
covid_max_date = date(2023, 5, 3)

# before YouTube removed dislikes (and base numeric feature set):
line_features = (
    SubplotFeature("view_count", "Views"),
    SubplotFeature("likes", "Likes"),
    SubplotFeature("dislikes", "Dislikes"),
    SubplotFeature("comment_count", "Comments"),
)
line_features_without_views = tuple(
    feature for feature in line_features if feature is not line_features[0]
)


# lines:
def line_1():
    # (1a) full dataset:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, line_features):
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
        )

    plt.show()

    # (1b) full dataset, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features:
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()

    # (1c) same as 1b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features_without_views:
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()


def line_2():
    # (2) within official COVID dates (pre):

    # could reuse:
    df_covid_mid = df_youtube[
        (df_youtube["trending_date"].dt.date >= covid_min_date)
        & (df_youtube["trending_date"].dt.date < covid_max_date)
    ]

    # (2a) pre-COVID:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, line_features):
        feature = feature_data.id
        df_trim = df_covid_mid[df_covid_mid[feature] != 0]

        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
        )

    plt.show()

    # (2b) pre-COVID, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features:
        feature = feature_data.id

        df_trim = df_covid_mid[df_covid_mid[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()

    # (2c) same as 2b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features_without_views:
        feature = feature_data.id

        df_trim = df_covid_mid[df_covid_mid[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()


def line_3():
    # (3) within official COVID dates (post):
    df_covid_post = df_youtube[df_youtube["trending_date"].dt.date >= covid_max_date]

    # (3a) post-COVID:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, line_features):
        feature = feature_data.id
        df_trim = df_covid_post[df_covid_post[feature] != 0]

        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        # since dislikes were removed, the isolated date range may provide no
        # groups, which will cause an error when calling zip:
        if len(groups_pairs) == 0:
            ax.set_visible(False)
            continue

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
        )

    plt.show()

    # (3b) post-COVID, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features:
        feature = feature_data.id

        df_trim = df_covid_post[df_covid_post[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        if len(groups_pairs) == 0:
            continue

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()

    # (3c) same as 3b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in line_features_without_views:
        feature = feature_data.id

        df_trim = df_covid_post[df_covid_post[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(map(lambda pair: (pair[0], pair[1][feature].sum()), groups))

        if len(groups_pairs) == 0:
            continue

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
        )

    ax.legend()
    plt.show()


def line_4():
    # (4) TODO -- rows with dislikes only?
    ...


def pie_1():
    # pie:
    pass


if __name__ == "__main__":
    line_1()
    line_2()
    line_3()
    # line_4()
    pie_1()
