"""
python_arXiv_paging_example.py

This sample script illustrates paging of arXiv api
results.  In order to play nice with the api, we
recommend that you wait 3 seconds between api calls.

Please see the documentation at
http://export.arxiv.org/api_help/docs/user-manual.html
for more information, or email the arXiv api
mailing list at arxiv-api@googlegroups.com.

urllib is included in the standard python library.
feedparser can be downloaded from http://feedparser.org/ .

Author: Julius B. Lucks

This is free software.  Feel free to do what you want
with it, but please play nice with the arXiv API!
"""

import time
import urllib

import arxiv2bib
import feedparser

# Base api query url
base_url = 'http://export.arxiv.org/api/query?';

# Search parameters
search_query = '"reinforcement+learning"+and+robot+and+transfer'  # search for electron in all fields
start = 0  # start at the first result
total_results = 306  # want 20 total results
results_per_iteration = 306  # 5 results at a time
wait_time = 1  # number of seconds to wait beetween calls

print('Searching arXiv for %s' % search_query)

arxiv_ids = []

for i in range(start, total_results, results_per_iteration):

    print("Results %i - %i" % (i, i + results_per_iteration))

    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, i, results_per_iteration)

    # perform a GET request using the base_url and query
    response = urllib.request.urlopen(base_url + query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    # Run through each entry, and print out information
    for entry in feed.entries:
        arxiv_id = entry.id.split('/abs/')[-1]
        arxiv_ids.append(arxiv_id)
    # Remember to play nice and sleep a bit before you call
    # the api again!
    print('Sleeping for %i seconds' % wait_time)
    time.sleep(wait_time)

bibtex = [f"{res.bibtex()}\n" for res in arxiv2bib.arxiv2bib(arxiv_ids)]


with open("result.bib", "w", encoding='utf-8') as fp:
    fp.writelines(bibtex)
