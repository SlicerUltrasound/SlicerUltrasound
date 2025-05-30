#!/usr/bin/env python3

import os
import sys
import json

def clean_nested_lines(lines):
    # Remove only empty sub-arrays, keep outer list even if empty
    return [line for line in lines if isinstance(line, list) and len(line) > 0]

def transform_annotations(data, rater):
    annotations = data.get("frame_annotations")
    if isinstance(annotations, dict):
        new_list = []
        for k in sorted(annotations.keys(), key=lambda x: int(x)):
            frame = annotations[k]
            if "annotation" in frame:
                frame.update(frame.pop("annotation"))

            pleura = clean_nested_lines(frame.get("pleura_lines", []))
            b_lines = clean_nested_lines(frame.get("b_lines", []))

            frame["pleura_lines"] = [
                {"rater": rater, "line": {"points": coords}} for coords in pleura
            ]
            frame["b_lines"] = [
                {"rater": rater, "line": {"points": coords}} for coords in b_lines
            ]

            frame["frame_number"] = int(k) if k.isdigit() else k
            frame["coordinate_space"] = "RAS"
            new_list.append(frame)

        data["frame_annotations"] = new_list
    return data

def extract_rater_from_path(path):
    last_segment = os.path.basename(os.path.normpath(path))
    if "-" in last_segment:
        return last_segment.split("-")[0].lower()
    return None

def convert_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        rater = extract_rater_from_path(root)
        if not rater:
            continue

        for file in files:
            if file.endswith(".json") and not file.startswith("._"):
                base_name = os.path.splitext(file)[0]
                out_file = f"{base_name}.{rater}.json"
                in_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_dir)
                out_path = os.path.join(output_dir, rel_dir, out_file)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                try:
                    with open(in_path, "r") as f:
                        data = json.load(f)

                    transformed = transform_annotations(data, rater)

                    with open(out_path, "w") as f:
                        json.dump(transformed, f, indent=2)

                    print(f"✅ Transformed and saved {out_file}")
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_annotations_rater.py <input_dir> <output_dir>")
        sys.exit(1)

    convert_directory(sys.argv[1], sys.argv[2])