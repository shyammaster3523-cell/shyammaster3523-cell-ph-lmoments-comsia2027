"""Download the two public datasets used in the paper into ./data.

AutoClaims : R package insuranceData (CRAN), https://CRAN.R-project.org/package=insuranceData
danishuni  : R package fitdistrplus (CRAN), https://CRAN.R-project.org/package=fitdistrplus
Files are fetched from the official read-only CRAN mirror on GitHub (github.com/cran).
"""
import os, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
FILES = {
    "AutoClaims.rda":  "https://raw.githubusercontent.com/cran/insuranceData/master/data/AutoClaims.rda",
    "x_danishuni.rda": "https://raw.githubusercontent.com/cran/fitdistrplus/master/data/danishuni.rda",
}
for name, url in FILES.items():
    dest = os.path.join(DATA, name)
    if not os.path.exists(dest):
        print("downloading", name); urllib.request.urlretrieve(url, dest)
    print(name, os.path.getsize(dest), "bytes")
