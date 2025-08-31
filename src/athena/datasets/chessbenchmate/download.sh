#!/bin/bash
BASE_URL="https://huggingface.co/datasets/joshuakgao/chessbenchmate/resolve/main"

# Download parts 000 through 117
for i in $(seq -w 038 117); do
    URL="${BASE_URL}/chessbenchmate.tar.part${i}"
    echo "Downloading ${URL}"
    wget "${URL}"
done

# Merge parts into one tar file
cat chessbenchmate.tar.part* > chessbenchmate.tar

# Extract the merged tar
tar -xvf chessbenchmate.tar
