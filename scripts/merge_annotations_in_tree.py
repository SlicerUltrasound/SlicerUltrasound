#!/usr/bin/env python3

import sys
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def merge_frame_annotations(file1_data, file2_data):
    frames1 = {str(f["frame_number"]): f for f in file1_data["frame_annotations"]}
    frames2 = {str(f["frame_number"]): f for f in file2_data["frame_annotations"]}
    merged = []

    all_frame_numbers = sorted(set(frames1.keys()).union(frames2.keys()), key=int)

    for fn in all_frame_numbers:
        frame = {}
        if fn in frames1:
            frame = json.loads(json.dumps(frames1[fn]))  # Deep copy
        if fn in frames2:
            f2 = frames2[fn]
            if not frame:
                frame = json.loads(json.dumps(f2))
            else:
                frame["pleura_lines"] = frame.get("pleura_lines", []) + f2.get("pleura_lines", [])
                frame["b_lines"] = frame.get("b_lines", []) + f2.get("b_lines", [])
        merged.append(frame)

    result = json.loads(json.dumps(file1_data))  # deep copy
    result["frame_annotations"] = merged
    return result

import os
from collections import defaultdict

def find_annotation_groups(root_dir):
    groups = defaultdict(list)
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json") and not filename.endswith(".combined.json"):
                parts = filename.rsplit(".", 2)
                if len(parts) == 3:
                    prefix, rater, ext = parts
                    groups[prefix].append(os.path.join(dirpath, filename))
    return groups

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_annotations_in_tree.py <input_root_directory> <output_root_directory>")
        sys.exit(1)

    input_root = sys.argv[1]
    output_root = sys.argv[2]
    annotation_groups = find_annotation_groups(input_root)

    copied_dcm_files = set()
    os.makedirs(output_root, exist_ok=True)

    for prefix, file_list in annotation_groups.items():
        logging.info(f"Merging files for prefix: {prefix}")
        merged_data = None
        sop_uid = None
        for file in file_list:
            logging.info(f"Processing input file: {file}")
            with open(file, "r") as f:
                data = json.load(f)
            if merged_data is None:
                merged_data = data
                sop_uid = data.get("SOPInstanceUID")
            else:
                if data.get("SOPInstanceUID") != sop_uid:
                    raise ValueError(f"SOPInstanceUID mismatch in file: {file}")
                merged_data = merge_frame_annotations(merged_data, data)

        output_path = os.path.join(output_root, prefix + ".combined.json")
        with open(output_path, "w") as out:
            json.dump(merged_data, out, indent=2)
        logging.info(f"✅ Merged output saved to {output_path}")

    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.endswith(".dcm") and fname not in copied_dcm_files:
                shutil.copy2(os.path.join(dirpath, fname), os.path.join(output_root, fname))
                copied_dcm_files.add(fname)
                logging.info(f"Copied .dcm file: {fname} from {dirpath}")