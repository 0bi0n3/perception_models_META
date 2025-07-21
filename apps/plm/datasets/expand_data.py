import os
import shutil

# --- Configuration ---
# Define the path to the annotations file relative to the project root
annotations_file = "apps/plm/datasets/ue_data/annotations.jsonl"
backup_file = annotations_file + ".bak"

# The file currently has 15 entries. We need to repeat them to get over 5000.
# 5000 / 15 = 333.33... -> We need 334 repetitions.
# 334 * 15 = 5010 total entries.
num_repeats = 334

# --- Script ---
print(f"Targeting annotations file: {annotations_file}")

# Create a backup of the original file if it doesn't exist
if not os.path.exists(backup_file):
    try:
        print(f"Creating a backup at: {backup_file}")
        shutil.copy(annotations_file, backup_file)
    except FileNotFoundError:
        print(f"Error: The file was not found at {annotations_file}")
        exit()
else:
    print(f"Backup file already exists at: {backup_file}")

# Read the original content of the file
try:
    with open(annotations_file, "r") as f:
        original_content = f.read()
except FileNotFoundError:
    print(f"Error: The file was not found at {annotations_file}")
    exit()

# Verify the file is not empty
if not original_content.strip():
    print("Error: The annotations file is empty. Nothing to repeat.")
    exit()

num_entries_original = original_content.count('}')
print(f"Found {num_entries_original} entries in the original file.")
print(f"Repeating the content {num_repeats} times...")

# Repeat the content
new_content = original_content * num_repeats

# Overwrite the file with the new, repeated content
with open(annotations_file, "w") as f:
    f.write(new_content)

num_entries_new = new_content.count('}')

print("\nDone!")
print(f"The file '{os.path.basename(annotations_file)}' has been updated.")
print(f"New number of entries: {num_entries_new}")
