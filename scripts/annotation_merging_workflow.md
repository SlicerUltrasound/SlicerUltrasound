# Annotation Merging Workflow

The scripts in `/scripts/` facilitate converting and preparing annotation files from multiple raters (annotators) so they can be merged and adjudicated in the `AdjudicateUltrasound` module.

## Complete Workflow

```
1. Multiple annotators use AnnotateUltrasound to create annotations:
   ├── tom/study1/3038953328.json
   └── sarah/study1/3038953328.json

2. Run `convert_to_annotated_format_with_rater.py`:
   └── output/
       ├── tom/study1/3038953328.tom.json      # with rater="tom"
       └── sarah/study1/3038953328.sarah.json  # with rater="sarah"

3. (Optional) Validate with `validate_annotations.py`:

4. Place all rater files + DICOM in same directory for adjudication:
   └── adjudication_folder/
       ├── 3038953328.dcm
       ├── 3038953328.tom.json
       └── 3038953328.sarah.json

5. `AdjudicateUltrasound` module loads and merges automatically:
   - Displays all lines from all raters
   - Color-codes by rater
   - Allows adjudicator to validate/invalidate each line
   - Saves adjudicated result as 3038953328.adjudication.json
```

## How Each Component Works

### 1. `convert_to_annotated_format_with_rater.py` - The Core Conversion Script

This is the **primary script** for transforming individual rater annotations into the merged format required by `AdjudicateUltrasound`.

### Input Format (from `AnnotateUltrasound`)
The `AnnotateUltrasound` module generates JSON files with `frame_annotations` as a **dictionary** keyed by frame number:

```json
{
  "frame_annotations": {
    "0": {
      "pleura_lines": [[[x1, y1, z1], [x2, y2, z2]]],
      "b_lines": [[[x1, y1, z1], [x2, y2, z2]]]
    }
  }
}
```

#### Output Format (for `AdjudicateUltrasound`)
The script transforms this to an **array-based format** with explicit rater attribution, matching the schema defined in `annotations.schema.json`:

```32:50:scripts/convert_to_annotated_format_with_rater.py
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
        # write out as LPS
        convert_ras_to_lps(data.get("frame_annotations", []))

    return data
```

#### Key Transformations:

1. **Rater Extraction** - Extracts rater name from directory path structure:

```52:57:scripts/convert_to_annotated_format_with_rater.py
def extract_rater_from_path(path):
    rater = os.path.basename(os.path.normpath(path))
    if rater:
        return rater.lower()
    else:
        return None
```

2. **Coordinate Conversion (RAS → LPS)** - Converts from Slicer's RAS coordinate system to LPS:

```12:21:scripts/convert_to_annotated_format_with_rater.py
def convert_ras_to_lps(annotations: list):
    for frame in annotations:
        if frame.get("coordinate_space", "RAS") == "RAS":
            for line_group in ["pleura_lines", "b_lines"]:
                for entry in frame.get(line_group, []):
                    points = entry["line"]["points"]
                    for point in points:
                        point[0] = -point[0]  # Negate X (Right → Left)
                        point[1] = -point[1]  # Negate Y (Anterior → Posterior)
            frame["coordinate_space"] = "LPS"  # Update coordinate_space
```

3. **Output Naming** - Creates rater-suffixed filenames:

```72:75:scripts/convert_to_annotated_format_with_rater.py
        rel_path = os.path.relpath(in_path, input_dir)
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        out_file = f"{base_name}.{rater}.json"
        out_path = os.path.join(output_dir, os.path.dirname(rel_path), out_file)
```

#### Expected Directory Structure

The script expects input organized as:
```
input_dir/
├── rater_name/
│   ├── subfolder/
│   │   └── study_id.json
```

And outputs:
```
output_dir/
├── rater_name/
│   ├── subfolder/
│   │   └── study_id.rater_name.json
```

---

### 2. How `AdjudicateUltrasound` Merges Annotations

The `AdjudicateUltrasound.loadNextSequence()` method performs the actual merging when loading a DICOM:

```1218:1254:AdjudicateUltrasound/AdjudicateUltrasound.py
            filepaths = glob.glob(f"{dir_path}/**/{base_name}.json", recursive=True) + glob.glob(f"{dir_path}/**/{base_name}.*.json", recursive=True)
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        ann = json.load(f)
                        self.convert_lps_to_ras(ann.get("frame_annotations", []))
                        # Merge non-frame_annotations keys with conflict check
                        for k, v in ann.items():
                            if k == "frame_annotations":
                                continue
                            # ... merging logic ...

                        # Only merge pleura_lines and b_lines for frames with matching frame_number
                        for frame in ann.get("frame_annotations", []):
                            frame_number = frame["frame_number"]
                            matched = next((f for f in merged_data["frame_annotations"] if f["frame_number"] == frame_number), None)
                            if matched:
                                matched["pleura_lines"].extend(frame.get("pleura_lines", []))
                                matched["b_lines"].extend(frame.get("b_lines", []))
                            else:
                                merged_data["frame_annotations"].append({
                                    "frame_number": frame["frame_number"],
                                    "coordinate_space": frame.get("coordinate_space", "RAS"),
                                    "pleura_lines": frame.get("pleura_lines", []),
                                    "b_lines": frame.get("b_lines", []),
                                })
```

The merging logic:
1. Searches for `{base_name}.json` and `{base_name}.*.json` files (e.g., `study.tom.json`, `study.sarah.json`)
2. Converts coordinates back from LPS → RAS for Slicer display
3. Deep-merges frame annotations by `frame_number`, combining `pleura_lines` and `b_lines` from all raters
4. Preserves the `rater` attribute on each line for adjudication

---

### 3. `validate_annotations.py` - Schema Validation

This script validates that annotation files conform to the expected schema:

```8:27:scripts/validate_annotations.py
def validate_directory(schema_path, json_root):
    with open(schema_path) as f:
        schema = json.load(f)

    failures = []

    def validate_file(full_path):
        try:
            with open(full_path) as f:
                data = json.load(f)

            validate(instance=data, schema=schema)
            print(f"✅ {full_path}")
        except (ValidationError, ValueError) as e:
            print(f"❌ {full_path} - {e}")
```

Usage:
```bash
python validate_annotations.py AnnotateUltrasound/annotations.schema.json /path/to/annotations/
```

---

### 4. Supporting Shell Scripts

#### `combine_all_files_from_tree.sh`
Flattens a directory tree by copying all `.dcm` and `.json` files to a single output directory:

```bash
./combine_all_files_from_tree.sh /path/to/source_dir /path/to/output_dir
```

Useful for collecting scattered annotation files before processing.

#### `copy_matching_dcm_tree.sh`
Copies `.dcm` files to match existing `.json` annotation files in a destination tree:

```bash
./copy_matching_dcm_tree.sh /path/to/source/dcm_dir /path/to/destination_root
```

This is useful when you have annotations but need to reunite them with their corresponding DICOM files for adjudication.

---

### Schema Format Reference

The target schema (`annotations.schema.json`) requires this structure:

```68:99:AnnotateUltrasound/annotations.schema.json
          "pleura_lines": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "rater": {
                  "type": "string"
                },
                "line": {
                  "type": "object",
                  "properties": {
                    "points": {
                      "type": "array",
                      "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                          "type": "number"
                        }
                      }
                    }
                  },
                  "required": [
                    "points"
                  ]
                }
              },
              "required": [
                "rater",
                "line"
              ]
            }
          },
```

Each line entry must have:
- `rater`: string identifying the annotator
- `line.points`: array of 3D coordinates `[[x, y, z], [x, y, z], ...]`