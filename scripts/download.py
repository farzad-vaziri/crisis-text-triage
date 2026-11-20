"""Download the Reddit Mental Health Dataset files used here (Zenodo 3941387, ODC-PDDL)."""
import urllib.request
from common import RAW, CLASSES, ZENODO

RAW.mkdir(parents=True, exist_ok=True)
for c in CLASSES:
    for period in ["2019", "post"]:
        f = f"{c}_{period}_features_tfidf_256.csv"
        dest = RAW / f
        if dest.exists():
            continue
        print("downloading", f)
        urllib.request.urlretrieve(ZENODO.format(f=f), dest)
print("done")
