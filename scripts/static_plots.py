import os
import json

from typing import NamedTuple
from math import floor, log10
from datetime import date, datetime

from scipy import stats
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import seaborn as sns

sns.set_style("white")
sns.set_color_codes("pastel")
sns.set_theme()

pd.set_option("display.float_format", "{:.2f}".format)
np.set_printoptions(precision=2)


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


def trim_feature_outliers(df: pd.DataFrame, feature_name: str):
    return df[(np.abs(stats.zscore(df[feature_name])) < 3)]


# modified impl., remove 0's (due to YouTube dislike removal) AND check for
# outliers in the same mask:
df_dislike_trim = df_youtube[
    (df_youtube["dislikes"] != 0) & (np.abs(stats.zscore(df_youtube["dislikes"])) < 3)
]


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
numeric_features = (
    SubplotFeature("view_count", "Views"),
    SubplotFeature("likes", "Likes"),
    SubplotFeature("dislikes", "Dislikes"),
    SubplotFeature("comment_count", "Comments"),
)
numeric_features_without_views = tuple(
    feature for feature in numeric_features if feature is not numeric_features[0]
)


def line_1():
    # (1a) full dataset:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, numeric_features):
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        # ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            linewidth=2,
        )

    plt.show()

    # (1b) full dataset, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features:
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
        )

    ax.legend()
    plt.show()

    # (1c) same as 1b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features_without_views:
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
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

    for ax, feature_data in zip(f.axes, numeric_features):
        feature = feature_data.id
        df_trim = df_covid_mid[df_covid_mid[feature] != 0]

        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        # ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            linewidth=2,
        )

    plt.show()

    # (2b) pre-COVID, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features:
        feature = feature_data.id

        df_trim = df_covid_mid[df_covid_mid[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
        )

    ax.legend()
    plt.show()

    # (2c) same as 2b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features_without_views:
        feature = feature_data.id

        df_trim = df_covid_mid[df_covid_mid[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
        )

    ax.legend()
    plt.show()


def line_3():
    # (3) within official COVID dates (post):
    df_covid_post = df_youtube[df_youtube["trending_date"].dt.date >= covid_max_date]

    # (3a) post-COVID:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, numeric_features):
        feature = feature_data.id
        df_trim = df_covid_post[df_covid_post[feature] != 0]

        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        # since dislikes were removed, the isolated date range may provide no
        # groups, which will cause an error when calling zip:
        if len(groups_pairs) == 0:
            ax.set_visible(False)
            continue

        ax.set_xlabel("Date")
        ax.set_ylabel(feature_data.display_name)
        # ax.yaxis.set_major_formatter(abbrev_num)

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            linewidth=2,
        )

    plt.show()

    # (3b) post-COVID, altogether:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features:
        feature = feature_data.id

        df_trim = df_covid_post[df_covid_post[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        if len(groups_pairs) == 0:
            continue

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
        )

    ax.legend()
    plt.show()

    # (3c) same as 3b., but without view_count:
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    # ax.yaxis.set_major_formatter(abbrev_num)

    for feature_data in numeric_features_without_views:
        feature = feature_data.id

        df_trim = df_covid_post[df_covid_post[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        if len(groups_pairs) == 0:
            continue

        ax.plot(
            tuple(zip(*groups_pairs))[0],
            tuple(zip(*groups_pairs))[1],
            label=feature_data.display_name,
            linewidth=2,
        )

    ax.legend()
    plt.show()


def line_4():
    # (4) TODO -- rows with dislikes only?
    ...


def bar_1():
    raise NotImplementedError


def count_1():
    # (1) numeric feature subplots, grouped by category [name]:
    groups = df_youtube[["category_name"]].groupby("category_name").size()

    # ax = plt.subplot()

    # ax.set_xlabel("Category")
    # ax.set_ylabel("Video Count")
    # ax.tick_params(axis="x", labelrotation=90)

    # bar_container = (tuple(groups.index), tuple(groups.values))
    # ax.bar_label(bar_container)

    ax = sns.barplot(x=tuple(groups.index), y=tuple(groups.values))
    ax.bar_label(ax.containers[0], fontsize=10)

    ax.tick_params(axis="x", labelrotation=90)

    plt.tight_layout()
    plt.show()


def pie_1():
    # (1) numeric feature subplots, grouped by category [name]:
    f, _ = plt.subplots(2, 2)

    for ax, feature_data in zip(f.axes, numeric_features):
        feature = feature_data.id
        df_trim = df_youtube[df_youtube[feature] != 0]

        groups = df_trim.groupby("category_name")
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )

        groups_top_n = sorted(groups_pairs, key=lambda pair: pair[1])[:5]

        if len(groups_pairs) == 0:
            ax.set_visible(False)
            continue

        ax.set_title(feature_data.display_name)
        ax.pie(
            tuple(zip(*groups_top_n))[1],
            labels=tuple(zip(*groups_top_n))[0],
            autopct="%.2f%%",
        )

    plt.show()


def dist_with_feature(feature_name: str):
    # outlier removal:
    sns.displot(trim_feature_outliers(df_youtube, feature_name), x=feature_name)
    plt.show()


def dist_1():
    # (1) dist plot w/ outlier removal, plotting view_count:
    dist_with_feature("view_count")


def dist_2():
    # (2) dist plot w/ outlier removal, plotting likes:
    dist_with_feature("likes")


def dist_3():
    # (3) dist plot w/ outlier removal, plotting dislikes:
    sns.displot(df_dislike_trim, x="dislikes")
    plt.show()


def dist_4():
    # (4) dist plot w/ outlier removal, plotting comment_count:
    dist_with_feature("comment_count")


def pair_1():
    sns.pairplot(
        df_youtube.sample(10000),
        vars=[
            feature.id
            for feature in numeric_features
            if feature is not numeric_features[2]
        ],
        hue="category_name",
        diag_kind="kde",
    )
    plt.show()


def heatmap_1():
    sns.heatmap(
        df_youtube[[feature.id for feature in numeric_features]].corr(),
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    plt.title("Views, Likes, Dislikes, and Comment Count Heatmap")

    plt.show()


def heatmap_1():
    df_trim = df_youtube[[feature.id for feature in numeric_features]]
    df_trim = df_trim[df_trim["dislikes"] != 0]
    # TODO -- figure out which one of these outlier operations to use:
    # df_trim = trim_feature_outliers(df_trim, "dislikes")

    sns.heatmap(
        df_trim.corr(),
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    plt.title("Views, Likes, Dislikes, and Comment Count Heatmap")

    plt.show()


def hist_kde_with_feature(feature_name: str):
    return sns.histplot(
        trim_feature_outliers(df_youtube, feature_name), x=feature_name, kde=True
    )


def hist_kde_1():
    # (1) hist with KDE for view_count
    ax = hist_kde_with_feature("view_count")
    ax.set_xlabel("Views")

    plt.show()


def hist_kde_2():
    # (2) hist with KDE for likes
    ax = hist_kde_with_feature("likes")
    ax.set_xlabel("Likes")

    plt.show()


def hist_kde_3():
    # (3) hist with KDE for dislikes
    ax = sns.histplot(df_dislike_trim, x="dislikes", kde=True)
    ax.set_xlabel("Dislikes")

    plt.show()


def hist_kde_4():
    # (3) hist with KDE for comment_count
    ax = hist_kde_with_feature("likes")
    ax.set_xlabel("Likes")

    plt.show()


def qq_1():
    raise NotImplementedError


def kde_with_feature(feature_name: str):
    return sns.kdeplot(
        trim_feature_outliers(df_youtube, feature_name),
        x=feature_name,
        linewidth=2,
        fill=True,
        alpha=0.6,
    )


def kde_1():
    # (1) KDE for view_count
    ax = kde_with_feature("view_count")
    ax.set_xlabel("Views")

    plt.show()


def kde_2():
    # (2) KDE for likes
    ax = kde_with_feature("likes")
    ax.set_xlabel("Likes")

    plt.show()


def kde_3():
    # (3) KDE for dislikes
    ax = sns.kdeplot(
        df_dislike_trim,
        x="dislikes",
        linewidth=2,
        fill=True,
        alpha=0.6,
    )
    ax.set_xlabel("Dislikes")

    plt.show()


def kde_4():
    # (4) KDE for comment_count
    ax = kde_with_feature("comment_count")
    ax.set_xlabel("Comments")

    plt.show()


def lm_1():
    # (1) lmplot views vs. likes: check to see if likes have increased with
    # respect to views
    grid = sns.lmplot(
        trim_feature_outliers(df_youtube, "view_count"),
        x="view_count",
        y="likes",
        line_kws={"color": "r"},
    )
    ax = grid.ax

    ax.set_title("lmplot: Views vs. Likes")
    ax.set_xlabel("Views")
    ax.set_ylabel("Likes")
    # ax.xaxis.set_major_formatter(abbrev_num)
    # ax.yaxis.set_major_formatter(abbrev_num)

    plt.show()


def lm_2():
    # (2) lmplot likes vs. comments: check to see if comments increase with
    # respect to likes
    grid = sns.lmplot(
        trim_feature_outliers(df_youtube, "likes"),
        x="likes",
        y="comment_count",
        line_kws={"color": "r"},
    )
    ax = grid.ax

    ax.set_title("lmplot: Comments vs. Likes")
    ax.set_xlabel("Likes")
    ax.set_ylabel("Comments")
    # ax.xaxis.set_major_formatter(abbrev_num)
    # ax.yaxis.set_major_formatter(abbrev_num)

    plt.show()


def lm_3():
    # (3) lmplot dislikes vs. comments: check to see if comments increase with
    # respect to dislikes
    grid = sns.lmplot(
        df_dislike_trim,
        x="dislikes",
        y="comment_count",
        line_kws={"color": "r"},
    )
    ax = grid.ax

    ax.set_title("lmplot: Comments vs. Likes")
    ax.set_xlabel("Dislikes")
    ax.set_ylabel("Comments")
    # ax.xaxis.set_major_formatter(abbrev_num)
    # ax.yaxis.set_major_formatter(abbrev_num)

    plt.show()


def boxplot_1():
    # (1) covers multiple fields: views, likes, dislikes, and comments; creates
    # and displays each plot individually

    # NOTE: HARDCODED! if you change numeric_features, change the second tuple!:
    feature_data_pairs = zip(
        numeric_features,
        (
            trim_feature_outliers(df_youtube, "view_count"),
            trim_feature_outliers(df_youtube, "likes"),
            df_dislike_trim,
            trim_feature_outliers(df_youtube, "comment_count"),
        ),
    )

    for feature_data, df in feature_data_pairs:
        feature = feature_data.id

        ax = plt.subplot()
        sns.boxplot(
            df,
            x=feature,
            y="category_name",
            ax=ax,
        )

        ax.set_xlabel(feature_data.display_name)
        ax.set_ylabel("Category")

        plt.show()


def boxplot_2():
    # (2) story: show dislikes next to views
    f, _ = plt.subplots(1, 2)

    # NOTE: HARDCODED! if you change numeric_features, change both tuples!:
    feature_data_pairs = zip(
        (numeric_features[2], numeric_features[0]),
        (
            df_dislike_trim,
            trim_feature_outliers(df_youtube, "view_count"),
        ),
    )

    for ax, (feature_data, df) in zip(f.axes, feature_data_pairs):
        feature = feature_data.id

        sns.boxplot(
            df,
            x=feature,
            y="category_name",
            ax=ax,
        )

        ax.set_xlabel(feature_data.display_name)
        ax.set_ylabel("Category")

    for ax in f.axes[1::2]:
        ax.set_ylabel("")
        ax.set_yticklabels([""] * len(ax.get_yticklabels()))

    plt.show()


def boxplot_3():
    # (2) story: show dislikes next to comments
    f, _ = plt.subplots(1, 2)

    # NOTE: HARDCODED! if you change numeric_features, change both tuples!:
    feature_data_pairs = zip(
        (numeric_features[2], numeric_features[3]),
        (
            df_dislike_trim,
            trim_feature_outliers(df_youtube, "comment_count"),
        ),
    )

    for ax, (feature_data, df) in zip(f.axes, feature_data_pairs):
        feature = feature_data.id

        sns.boxplot(
            df,
            x=feature,
            y="category_name",
            ax=ax,
        )

        ax.set_xlabel(feature_data.display_name)
        ax.set_ylabel("Category")

    for ax in f.axes[1::2]:
        ax.set_ylabel("")
        ax.set_yticklabels([""] * len(ax.get_yticklabels()))

    plt.show()


def area_with_features(feature_set: list[SubplotFeature]):
    ax = plt.subplot()

    ax.grid()
    ax.set_xlabel("Dates")
    ax.set_ylabel("Count")

    for feature_data in feature_set:
        feature = feature_data.id

        df_trim = df_youtube[df_youtube[feature] != 0]
        groups = df_trim.groupby([df_trim["trending_date"].dt.date])
        groups_pairs = list(
            map(lambda pair: (pair[0][0], pair[1][feature].sum()), groups)
        )
        # ax.yaxis.set_major_formatter(abbrev_num)

        x = tuple(zip(*groups_pairs))[0]
        y = tuple(zip(*groups_pairs))[1]

        ax.plot(
            x,
            y,
            label=feature_data.display_name,
            linewidth=2,
        )
        ax.fill_between(x, y, alpha=0.6)

    ax.legend()

    return ax


def area_1():
    area_with_features([*numeric_features[:2], *numeric_features[3:]])
    plt.show()


def area_2():
    area_with_features([numeric_features[1], numeric_features[3]])
    plt.show()


def violin_1():
    # (2) violin plot w/ trimmed views dataset
    ax = sns.violinplot(
        trim_feature_outliers(df_youtube, "view_count"),
        x="view_count",
        y="category_name",
    )

    ax.set_xlabel("Views")
    ax.set_ylabel("Category")

    plt.show()


def violin_2():
    # (2) violin plot w/ trimmed likes dataset
    ax = sns.violinplot(
        trim_feature_outliers(df_youtube, "likes"),
        x="likes",
        y="category_name",
    )

    ax.set_xlabel("Likes")
    ax.set_ylabel("Category")

    plt.show()


def violin_3():
    # (3) violin plot w/ dislikes (using pre-filtered set)
    ax = sns.violinplot(
        df_dislike_trim,
        x="dislikes",
        y="category_name",
    )

    ax.set_xlabel("Dislikes")
    ax.set_ylabel("Category")

    plt.show()


def violin_4():
    # (4) violin plot w/ trimmed comments dataset
    ax = sns.violinplot(
        trim_feature_outliers(df_youtube, "comment_count"),
        x="comment_count",
        y="category_name",
    )

    ax.set_xlabel("Comments")
    ax.set_ylabel("Category")

    plt.show()


def joint_with_features(feature_x: str, feature_y: str):
    df_filtered = df_youtube[[feature_x, feature_y, "category_name"]][
        (np.abs(stats.zscore(df_youtube)) < 3).all(axis=1)
    ]

    return sns.jointplot(df_filtered, feature_x, feature_y, hue="category_name")


def joint_1():
    joint_with_features("likes", "comment_count")
    plt.show()


def rug_with_features(df: pd.DataFrame, feature_x: str, feature_y: str):
    sns.scatterplot(df, x=feature_x, y=feature_y)
    return sns.rugplot(df, x=feature_x, y=feature_y)


def rug_1():
    ax = rug_with_features(df_youtube, "likes", "comment_count")
    
    ax.set_xlabel("Likes")
    ax.set_ylabel("Comments")

    plt.show()

def rug_2():
    ax = rug_with_features(df_dislike_trim, "dislikes", "comment_count")
    
    ax.set_xlabel("Dislikes")
    ax.set_ylabel("Comments")

    plt.show()

def rug_3():
    ax = rug_with_features(df_dislike_trim, "dislikes", "view_count")

    ax.set_xlabel("Dislikes")
    ax.set_ylabel("Views")

    plt.show()


if __name__ == "__main__":
    # line:
    # line_1()
    # line_2()
    # line_3()
    # line_4()

    # bar: TODO
    # bar_1() # grouped
    # bar_2() # stacked

    # count:
    # count_1()

    # pie:
    # pie_1()

    # dist:
    # dist_1()
    # dist_2()
    # dist_3()
    # dist_4()

    # pair:
    # pair_1()

    # heatmap:
    # heatmap_1()

    # histogram w/ kde:
    # hist_kde_1()
    # hist_kde_2()
    # hist_kde_3()
    # hist_kde_4()

    # qq: TODO
    # qq_1()

    # KDE:
    # kde_1()
    # kde_2()
    # kde_3()
    # kde_4()

    # lmplot:
    # lm_1()
    # lm_2()
    # lm_3()

    # boxplot:
    # boxplot_1()
    # boxplot_2()
    # boxplot_3()

    # area:
    # area_1()
    # area_2()

    # violin:
    # violin_1()
    # violin_2()
    # violin_3()
    # violin_4()

    # joint: FIXME: too slow!
    # joint_1()

    # rug:
    # rug_1()
    # rug_2()
    # rug_3()
    ...
