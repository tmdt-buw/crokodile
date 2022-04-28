import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
import os
from collections import Counter
import pandas as pd
import seaborn as sns

corpus_file = "C:/Users/christian/Documents/Dissertation/Publications/0_in_progress/A Survey of Cross-Domain Transfer Learning in Robotics/corpus/corpus_relevant.bib"

consolidated_bib = BibDatabase()

existing_titles = set()
existing_dois = set()

pd.read_csv()

with open(corpus_file, "r", encoding="utf-8") as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

print(len(bib_database.entries))

words_in_titles = []
words_in_abstracts = []
blacklist = frozenset({'', 'using', 'those', 'on', 'own', 'yourselves', 'ie', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'cry', 'hereafter', 'front', 'too', 'wherein', 'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'cant', 'was', 'con', 'have', 'into', 'because', 'inc', 'not', 'therefore', 'they', 'even', 'whom', 'it', 'see', 'somewhere', 'interest', 'thereupon', 'nothing', 'thick', 'whereas', 'much', 'whenever', 'find', 'seem', 'until', 'whereby', 'at', 'ltd', 'fire', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'once', 'will', 'noone', 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'etc', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'thin', 'ourselves', 'few', 'third', 'without', 'anything', 'twelve', 'against', 'while', 'twenty', 'if', 'however', 'found', 'herself', 'when', 'may', 'ours', 'six', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'fill', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'eg', 'others', 'show', 'sincere', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'de', 'latterly', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'could', 'five', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'here', 'during', 'why', 'with', 'becomes', 'about', 'a', 'co', 'seeming', 'due', 'wherever', 'beforehand', 'detail', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', 'top', 'her', 'nobody', 'sometime', 'across', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'describe', 'under', 'always', 'himself', 'in', 'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them', 'hasnt', 'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has', 'yours', 'whose', 'behind', 'please', 'amoungst', 'mill', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'same', 'rather', 'latter', 'and', 'hereupon', 'part', 'per', 'eleven', 'ever', 'enough', 'again', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', 'give', 'system', 'do', 'an', 'these', 'everyone', 'towards', 'this', 'bill', 'cannot', 'un', 'afterwards', 'beyond', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'serious', 're', 'two', 'couldnt', 'less'})

for entry in bib_database.entries:
    words_in_titles += [word.lower() for word in entry["title"].split(" ")]
    words_in_abstracts += [word.lower() for word in entry["abstract"].split(" ")]

words_in_titles = [w for w in words_in_titles if w not in blacklist]
words_in_abstracts = [w for w in words_in_abstracts if w not in blacklist]

print(Counter(words_in_titles))
print(Counter(words_in_abstracts))

counter = Counter()

for word in set(words_in_titles):
    for entry in bib_database.entries:
        if word in entry["title"].lower():
            counter[word] += 1

print(counter)

words_in_titles = [x[0] for x in counter.most_common(10)]
words_in_titles.append("reinforcement learning")

print(words_in_titles)

df = pd.DataFrame(index=[entry["ID"] for entry in bib_database.entries])

for entry in bib_database.entries:
    for word_in_title in words_in_titles:
        df.loc[entry["ID"], "t_" + word_in_title] = int(word_in_title in entry["title"].lower())
        
    # for word_in_abstract in words_in_abstracts:
    #     df.loc[entry["ID"], "a_" + word_in_abstract] = int(word_in_abstract in entry["abstract"].lower())

# for column in df.columns:
#     if df[column].sum() < 5:
#         df = df.drop(column, axis=1)

df_combined_columns = pd.DataFrame(index=df.columns, columns=df.columns)

for columnA in df.columns:
    for columnB in df.columns:
        combined_sum = sum(df[columnA].astype(bool) | df[columnB].astype(bool))
        df_combined_columns.loc[columnA,columnB] = combined_sum

df_combined_columns.to_csv("combined_features.csv", sep=";")
df.to_csv("keywords.csv", sep=";")

words_in_abstracts = [x[0] for x in Counter(words_in_abstracts).most_common(10)]

entries_filtered = [entry for entry in bib_database.entries if "transfer" not in entry["title"].lower() and "imitation" not in entry["title"].lower()]

for entry in entries_filtered:
    words_in_abstracts += [word.lower() for word in entry["abstract"].split(" ")]

words_in_abstracts = [w for w in words_in_abstracts if w not in blacklist]
words_in_abstracts = [x[0] for x in Counter(words_in_abstracts).most_common(10)]

print(words_in_abstracts)

df2 = pd.DataFrame(index=[entry["ID"] for entry in entries_filtered])

for entry in entries_filtered:
    for word_in_abstract in words_in_abstracts:
        df2.loc[entry["ID"], "a_" + word_in_abstract] = int(word_in_abstract in entry["abstract"])

df2.to_csv("abstract.csv", sep=";")

    #
    # print(Counter(["paper" if entry["ENTRYTYPE"] in ["article", "inproceedings", "inbook"] and "author" in entry else entry["ENTRYTYPE"] for entry in bib_database.entries]))
    #
    # paper_entries = [entry for entry in bib_database.entries if entry["ENTRYTYPE"] in ["article", "inproceedings", "inbook"] and "author" in entry]

    # for entry in paper_entries:
    #     # if "proceedings" in entry["title"].lower():
    #     print(entry["title"])
        # if entry["ENTRYTYPE"] not in ["inproceedings", "proceedings"]:
        #     # print(f"Skip (no author): {entry.get('title', '')}")
        #     print(entry["ENTRYTYPE"])