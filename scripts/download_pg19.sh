#!/usr/bin/env bash
# Download the PG-19 books published in or before 1902.
#
#   scripts/download_pg19.sh [DATA_DIR]      (default: /home/user/data)
#
# Fetches PG-19's metadata.csv (id, title, publication_date, url; no header
# row), keeps books with publication_date <= 1902 (a year-only 1903 could be
# after 17 Dec 1903, so it is excluded), finds each book in the train,
# validation or test split of gs://deepmind-gutenberg, and copies them with
# `gsutil -m cp -I` into DATA_DIR/pg19_raw/<id>.txt. Expected: 17,350 files,
# about 6.8 GB. Re-running skips files already present (cp -n).
set -euo pipefail

DATA_DIR="${1:-/home/user/data}"
BUCKET="gs://deepmind-gutenberg"
RAW="$DATA_DIR/pg19_raw"
mkdir -p "$RAW"
cd "$DATA_DIR"

curl -sSf -o metadata.csv "https://storage.googleapis.com/deepmind-gutenberg/metadata.csv"

for split in train validation test; do
    gsutil ls "$BUCKET/$split/" > "ls_$split.txt"
done

python3 - <<'EOF'
import csv

wanted = set()
with open("metadata.csv", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) >= 3 and row[2].strip().isdigit() and int(row[2]) <= 1902:
            wanted.add(row[0])

urls = []
for split in ("train", "validation", "test"):
    with open(f"ls_{split}.txt") as f:
        for line in f:
            url = line.strip()
            if url.endswith(".txt") and url.rsplit("/", 1)[1][:-4] in wanted:
                urls.append(url)

with open("download_list.txt", "w") as f:
    f.write("\n".join(urls) + "\n")
print(f"{len(wanted)} books dated <= 1902; {len(urls)} found in the bucket")
EOF

gsutil -m cp -n -I "$RAW/" < download_list.txt 2> gsutil.log && status=0 || status=$?
echo "EXIT $status" >> gsutil.log
echo "$(ls "$RAW" | wc -l) files in $RAW (gsutil exit $status)"
exit "$status"
