import pandas as pd

df_all = pd.read_excel("all_collections.xlsx")
df_wos = pd.read_excel("amc.xlsx")

print(df_all.columns)
print(df_wos.columns)

keys = ['Titel']

i1 = df_all.set_index(keys).index
i2 = df_wos.set_index(keys).index

df_contained_in_all = df_all[i1.isin(i2)]
df_not_contained_in_all = df_all[~i1.isin(i2)]

print(df_contained_in_all.shape)
print(df_not_contained_in_all.shape)

exit()


print(df_all.shape)
print(df_wos.shape)
print(df_arxiv_acm.shape)