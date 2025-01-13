#!/bin/bash

# Path to the text file containing the file paths
FILE_LIST="/home/schmidtfa/experiments/resting_tinnitus/all_paths.txt"

# Base destination directory
DEST_DIR="/home/experiments/resting_tinnitus/data"

# Read each line from the file list
while IFS= read -r line; do
    # Remove the single quotes from the file path
    FILE_PATH=${line:1:-1}

    # Construct the new destination path by replacing the old base path with the new base directory
    NEW_DEST_PATH=${FILE_PATH/\/mnt\/sinuhe\/data_raw/$DEST_DIR}

    # Create the destination directory structure
    mkdir -p "$(dirname "$NEW_DEST_PATH")"

    # Copy the file using rsync
    rsync -avz b1059770@obob-bomber-fschmidt.hpc.sbg.ac.at:"$FILE_PATH" "$NEW_DEST_PATH" 
done < "$FILE_LIST"

