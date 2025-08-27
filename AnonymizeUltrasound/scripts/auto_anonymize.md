# *auto_anonymize.py* – Command-line Ultrasound-DICOM Anonymizer

Replicates the 3D Slicer AnonymizeUltrasound extension entirely from the shell, automating fan-masking and removing PHI from the DICOM headers.

⸻

1. Quick start

First, install the dependencies.
```python
uv venv --python 3.9.10
source .venv/bin/activate
uv pip install -r requirements-cpu.txt
```

To run the script, use the following command:
```
python -m auto_anonymize \
    /path/to/input_dicoms \
    /path/to/output_dicoms \
    /path/to/headers_out \
    --model-path /path/to/model_trace.pt \
    --device cuda \
    --overview-dir /tmp/overviews
```

At the end you will have:

- /path/to/output_dicoms	Fully-anonymized DICOMs (same tree as input unless you turn off preservation).
- /path/to/headers_out/keys.csv	Lookup table mapping original → anonymized filenames / UIDs.
- /path/to/headers_out/*_DICOMHeader.json	JSON copy of every header (patient name & DOB stripped).
- /tmp/overviews	Side-by-side PNGs of original vs anonymized first frames, for manual QC.


⸻

2. Required arguments

| Positional arg | Purpose |
| --------------- | ------- |
| input_folder | Root directory to scan for DICOM files (recursively). Only files with Modality == "US" are processed. |
| output_folder | Where anonymized DICOMs are written. Sub-folders are copied 1-to-1 unless you disable that (see below). |
| headers_folder | Separate store for headers/keys and the keys.csv de-identification map. Often in the same root as output_folder. |
| model_path | *.pt checkpoint for the Attention U-Net + DSNT corner-regression model. |

⸻

3. Optional switches

| Flag | Default | What it does |
| --------------- | ------- | ------- |
| --device {cpu,cuda,mps} | cpu | Where inference runs. Falls back to CPU if the requested accelerator (MPS, CUDA) is missing. |
| --skip-single-frame | off | Ignores single-frame studies that don’t need masking. |
| --no-hash-patient-id | off | **Dangerous** – keeps original PatientID instead of hashing the first 10 digits. |
| --filename-prefix PREFIX | none | Prepends PREFIX_ to every output filename. Helpful when wanting to mark the source of the anonymized data. |
| --no-preserve-directory-structure | off | Dumps all output files flat into output_folder instead of mirroring the input tree. |
| --resume-anonymization | off | If the final .dcm already exists, the file is skipped; useful for interrupted runs. |
| --overview-dir DIR | none | Saves PNG grids (original vs masked) for the first frame of each clip. **Highly recommended for PHI QC.** |


⸻

4. What actually happens

	1.	Directory scan
Builds a Pandas dataframe of every ultrasound DICOM, gathering UIDs, frame count, spacing, etc.
	2.	Filename & key generation: `<10-digit hash(PatientID)>_<8-digit hash(SOPInstanceUID)>.dcm`. The mapping is written to keys.csv.

    3. Inference
        - preprocessing.lib.create_frames.read_frames_from_dicom extracts frames as N×C×H×W (uint8).
        - The Attention U-Net + DSNT model predicts four normalized corner points, denormalized to pixel space.
        - Mask creation & application
        - Converted to a fan-shaped binary mask via compute_masks_and_configs; mask is multiplied into every frame.
	5.	DICOM re-assembly
        - Pixel data re-encapsulated as JPEG baseline.
        - Mandatory tags copied (BitsAllocated, Manufacturer, …).
        - Fresh SeriesInstanceUID generated; other UIDs preserved unless missing.
        - Dates randomly shifted ≤30 days (consistent per patient).
        - Patient name/ID cleared (or hashed) and birth date truncated to year only.
	6.	Outputs written
        - .dcm, matching .json sequence info (mask metadata) and header JSON.
        - Optional PNG overview.

Progress is shown with a TQDM bar and detailed timing blocks (read, inference, mask, save, etc.) in the log.

⸻

5. Logging & exit codes

| Outcome | Log location | Exit status |
| ------- | ------------ | ----------- |
| All files succeed | auto_anonymize_*.log (created by utils.logging_utils) | 0 |
| ≥ 1 file fails | Log highlights the failures | 1 |

Pass --resume-anonymization to re-run later without re-processing finished files.
