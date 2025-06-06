#!/bin/bash

set -e

# Usage: ./copy_matching_dcm_tree.sh /path/to/source/dcm_dir /path/to/destination_root
src_dir="$1"
dest_root="$2"

if [[ ! -d "$src_dir" || ! -d "$dest_root" ]]; then
  echo "Usage: $0 /path/to/source/dcm_dir /path/to/destination_root"
  exit 1
fi

find "$dest_root" -type f -name "*.json" | while read -r json_file; do
  prefix=$(basename "$json_file" | cut -d. -f1)
  target_dir=$(dirname "$json_file")
  dcm_file="$src_dir/$prefix.dcm"

  if [[ -f "$dcm_file" ]]; then
    echo "Copying $dcm_file → $target_dir/"
    cp "$dcm_file" "$target_dir/"
  fi
done
