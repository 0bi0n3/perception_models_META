#!/bin/bash

# This script copies a specified checkpoint directory to a target location,
# typically a scratch space for backup or further use.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---

# The directory of the checkpoint you want to copy.
# This should be the folder containing the 'consolidated.pth' and 'params.json' files.
SOURCE_CHECKPOINT_DIR="checkpoints/plm_3b_pe_finetune/checkpoints/0000000005/"

# The base directory in your scratch space where checkpoints will be stored.
SCRATCH_CHECKPOINTS_DIR="/mnt/hitchcock/scratch/oberon/checkpoints/"

# --- Script Logic ---

echo "--- Checkpoint Copy Script ---"

# 1. Check if the source directory exists
if [ ! -d "$SOURCE_CHECKPOINT_DIR" ]; then
    echo "Error: Source directory '$SOURCE_CHECKPOINT_DIR' not found."
    echo "Please ensure you are running this script from the 'perception_models' project root and the path is correct."
    exit 1
fi

# 2. Create the destination directory
# This will create the base directory and a subdirectory for this specific checkpoint.
CHECKPOINT_NAME=$(basename "$SOURCE_CHECKPOINT_DIR")
DESTINATION_DIR="${SCRATCH_CHECKPOINTS_DIR}${CHECKPOINT_NAME}/"

echo "Creating destination directory: $DESTINATION_DIR"
mkdir -p "$DESTINATION_DIR"

# 3. Copy the files
echo "Copying files from '$SOURCE_CHECKPOINT_DIR' to '$DESTINATION_DIR'..."
cp -rv "$SOURCE_CHECKPOINT_DIR"* "$DESTINATION_DIR"

echo "---------------------------------"
echo "✅ Copy complete!"
echo "Checkpoint is now available at: $DESTINATION_DIR"