import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_keywords = pd.read_csv("keywords.csv", sep=";", index_col=0)
df_keywords = df_keywords.astype(bool)
df_keywords = df_keywords.iloc[:,:-1]

df_combined = pd.DataFrame(index=df_keywords.columns, columns=df_keywords.columns)

for iA, columnA in enumerate(df_keywords.columns):
    for iB, columnB in enumerate(df_keywords.columns):
        if iA >= iB:
            combined_sum = sum(df_keywords[columnA] | df_keywords[columnB])
        else:
            combined_sum = None
        df_combined.loc[columnA,columnB] = combined_sum

df_combined.columns = df_combined.columns.str.replace("t_", "")
df_combined.index = df_combined.index.str.replace("t_", "")
sns.heatmap(df_combined.astype(float), annot=True, fmt=".0f", cbar=False, vmin=0, vmax=31)
plt.title("Title keywords | OR linkage")
plt.tight_layout()
plt.show()

df_combined = pd.DataFrame(index=df_keywords.columns, columns=df_keywords.columns)

for iA, columnA in enumerate(df_keywords.columns):
    for iB, columnB in enumerate(df_keywords.columns):
        if iA >= iB:
            combined_sum = sum(df_keywords[columnA] & df_keywords[columnB])
        else:
            combined_sum = None
        df_combined.loc[columnA,columnB] = combined_sum

df_combined.columns = df_combined.columns.str.replace("t_", "")
df_combined.index = df_combined.index.str.replace("t_", "")
sns.heatmap(df_combined.astype(float), annot=True, fmt=".0f", cbar=False, vmin=0, vmax=31)
plt.title("Title keywords | AND linkage")
plt.tight_layout()
plt.show()

df_combined = pd.DataFrame(index=df_keywords.columns, columns=df_keywords.columns)

for iA, columnA in enumerate(df_keywords.columns):
    for iB, columnB in enumerate(df_keywords.columns):
        if iA > iB:
            combined_sum = sum(df_keywords[columnA] ^ df_keywords[columnB])
        else:
            combined_sum = None
        df_combined.loc[columnA,columnB] = combined_sum

df_combined.columns = df_combined.columns.str.replace("t_", "")
df_combined.index = df_combined.index.str.replace("t_", "")
sns.heatmap(df_combined.astype(float), annot=True, fmt=".0f", cbar=False, vmin=0, vmax=31)
plt.title("Title keywords | XOR linkage")
plt.tight_layout()
plt.show()

df_keywords[~df_keywords["t_transfer"] & ~df_keywords["t_imitation"]]

exit()

import plotly.graph_objects as go

total = 1164

amc = 743
amc_collection = 241
amc_article = 502

wos = 161
wos_article = 158
wos_collection = 2

arxiv = 260  #

relevant_title_abstract = 54

relevant = 24

print(amc_article + wos_article + arxiv)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        # line=dict(color="black", width=0.5),
        label=["AMC (502 paper)", "WOS (158 paper)", "arXiv (260 paper)", "920 paper", "54 paper",
               "24 paper"],
        x=[0., 0., 0., 1, 1, 2, 3],
        y=[0, 1, 2, 0, 0, 0, 0],
        # color="blue"
    ),
    link=dict(
        source=[0, 1, 2, 3, 4],  # indices correspond to labels, eg A1, A2, A1, B1, ...
        target=[3, 3, 3, 4, 5],
        value=[amc_article, wos_article, arxiv, relevant_title_abstract, relevant]
    ))])

column_names = ["Data Sources", "Step 1: Data Collection", "Raw Corpus", "Step 2: Abstract Screening", "Preliminary Corpus",
                "Step 3: Full-Paper Screening",
                "Final Corpus"]
x_coordinates = [0, .5, 1, 1.5, 2, 2.5, 3]
y_coordinates = [-.05, 1.05, -.05, 1.05, -.05, 1.05, -.05]

for x_coordinate, y_coordinate, column_name in zip(x_coordinates, y_coordinates, column_names):
    fig.add_annotation(x=x_coordinate, y=y_coordinate, xref="x", yref="paper", text=column_name, showarrow=False,
                       font=dict(size=20, ), align="center")

fig.update_layout(
    title_text="Basic Sankey Diagram",
    xaxis={
        'showgrid': False,  # thin lines in the background
        'zeroline': False,  # thick line at x=0
        'visible': False,  # numbers below
    },
    yaxis={
        'showgrid': False,  # thin lines in the background
        'zeroline': False,  # thick line at x=0
        'visible': False,  # numbers below
    }, plot_bgcolor='rgba(0,0,0,0)', font_size=30)

# fig.write_image("corpus.svg")

fig.show()
