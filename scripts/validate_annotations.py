#!/usr/bin/env python3

import os
import sys
import json
from jsonschema import validate, ValidationError

def validate_directory(schema_path, json_root):
    with open(schema_path) as f:
        schema = json.load(f)

    failures = []
    for root, _, files in os.walk(json_root):
        for file in files:
            if file.endswith(".json") and not file.startswith("._"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path) as f:
                        data = json.load(f)

                    # Validate against schema only
                    validate(instance=data, schema=schema)
                    print(f"✅ {full_path}")
                except (ValidationError, ValueError) as e:
                    print(f"❌ {full_path} - {e}")
                    failures.append((full_path, str(e)))
                except Exception as e:
                    print(f"⚠️ {full_path} - Failed to read or parse: {e}")
                    failures.append((full_path, str(e)))

    print(f"\nDone. {len(failures)} file(s) failed validation.")
    return failures

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_annotations.py <schema_file> <directory>")
        sys.exit(1)

    schema_file = sys.argv[1]
    target_dir = sys.argv[2]
    validate_directory(schema_file, target_dir)