#!/usr/bin/env bash
# Download and extract the PhaBOX v2 database.
# Run once before starting the service.
set -euo pipefail

DB_URL="https://github.com/KennthShang/PhaBOX/releases/download/v2/phabox_db_v2_1.zip"
DB_DIR="phabox_db_v2"

if [ -d "$DB_DIR" ]; then
    echo "Database directory '$DB_DIR' already exists. Skipping download."
    exit 0
fi

echo "Downloading PhaBOX database..."
wget -q --show-progress "$DB_URL" -O phabox_db_v2_1.zip

echo "Extracting..."
unzip -q phabox_db_v2_1.zip
rm phabox_db_v2_1.zip

echo "Done. Database extracted to '$DB_DIR'."
