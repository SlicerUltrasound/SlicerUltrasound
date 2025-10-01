#!/bin/bash

# batch_auto_anonymize.sh
# Script to batch process nested DICOM directories using auto_anonymize.py
# Usage: ./batch_auto_anonymize.sh <input_root_dir> <output_root_dir> <headers_root_dir> <mode> [options]

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables set by parse_arguments
INPUT_ROOT=""
OUTPUT_ROOT=""
HEADERS_ROOT=""
MODE=""
OVERVIEW_ROOT=""
MODEL_PATH=""
SCRIPT_DIR=""

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << 'EOF'
Usage: ./batch_auto_anonymize.sh <input_root_dir> <output_root_dir> <headers_root_dir> <mode> [options]

Required Arguments:
  input_root_dir    Root directory to scan for DICOM files
  output_root_dir   Directory for anonymized DICOMs
  headers_root_dir  Directory for headers/keys
  mode              Anonymization mode: 'full' or 'phi-only'

Optional Arguments:
  [overview_root_dir]              Directory for QC overview images (positional, for backward compatibility)
  --overview-dir <path>            Directory for QC overview images (only for 'full' mode)
  --model-path <path>              Path to model checkpoint file
                                   Default: ~/path/to/model.pt
  --script-dir <path>              Path to scripts directory
                                   Default: ~/path/to/scripts

Modes:
  full      - Full anonymization with fan masking
              Optionally generates before/after comparison images
  phi-only  - Only anonymize PHI region at top of image
              Uses --phi_only_mode and --remove_phi_from_image flags

Examples:
  # Full anonymization with overview images (positional)
  ./batch_auto_anonymize.sh /data/input /data/output /data/headers full /data/overview

  # Full anonymization with named arguments
  ./batch_auto_anonymize.sh /data/input /data/output /data/headers full --overview-dir /data/overview

  # PHI-only mode with custom model path
  ./batch_auto_anonymize.sh /data/input /data/output /data/headers phi-only --model-path /custom/path/model.pt

  # Full example with all custom paths
  ./batch_auto_anonymize.sh /data/input /data/output /data/headers full \
     --overview-dir /data/overview \
     --model-path /custom/model.pt \
     --script-dir /custom/scripts

This script will:
  1. Find all directories containing DICOM files (.dcm, .dicom)
  2. Process each directory with auto_anonymize.py
  3. Preserve the directory structure in output locations
  4. Use CPU processing to avoid memory issues
  5. Apply the selected anonymization mode
EOF
}

# Function to parse arguments
parse_arguments() {
    # Required positional arguments
    if [ $# -lt 4 ]; then
        show_usage
        exit 1
    fi

    INPUT_ROOT="$1"
    OUTPUT_ROOT="$2"
    HEADERS_ROOT="$3"
    MODE="$4"

    # Optional positional and named arguments
    shift 4
    OVERVIEW_ROOT=""

    # Parse remaining arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --model-path)
                if [ $# -lt 2 ]; then
                    log_error "--model-path requires a value"
                    exit 1
                fi
                MODEL_PATH="$2"
                shift 2
                ;;
            --script-dir)
                if [ $# -lt 2 ]; then
                    log_error "--script-dir requires a value"
                    exit 1
                fi
                SCRIPT_DIR="$2"
                shift 2
                ;;
            --overview-dir)
                if [ $# -lt 2 ]; then
                    log_error "--overview-dir requires a value"
                    exit 1
                fi
                OVERVIEW_ROOT="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                # Assume it's the overview_dir (for backward compatibility)
                if [ -z "$OVERVIEW_ROOT" ]; then
                    OVERVIEW_ROOT="$1"
                else
                    log_error "Unknown argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# Function to check if directory contains DICOM files
has_dicom_files() {
    local dir="$1"
    # Check for .dcm or .dicom files (case insensitive)
    find "$dir" -maxdepth 1 -type f \( -iname "*.dcm" -o -iname "*.dicom" \) | head -1 | grep -q . 2>/dev/null
}

# Function to count DICOM files in directory
count_dicom_files() {
    local dir="$1"
    find "$dir" -maxdepth 1 -type f \( -iname "*.dcm" -o -iname "*.dicom" \) | wc -l
}

# Function to get memory usage in MB
get_memory_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ps -o rss= -p $$ | awk '{print int($1/1024)}'
    else
        # Linux
        ps -o rss= -p $$ | awk '{print int($1/1024)}'
    fi
}

# Function to process a single directory
process_directory() {
    local input_dir="$1"
    local output_dir="$2"
    local headers_dir="$3"
    local relative_path="$4"
    local mode="$5"
    local overview_dir="$6"  # May be empty
    local model_path="$7"
    local script_dir="$8"

    local dicom_count=$(count_dicom_files "$input_dir")
    local mem_before=$(get_memory_usage)

    log_info "Processing: $relative_path ($dicom_count DICOM files) [Mode: $mode]"
    log_info "Memory usage before: ${mem_before}MB"

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$headers_dir"

    # Create overview directory if specified
    if [ -n "$overview_dir" ]; then
        mkdir -p "$overview_dir"
    fi

    # Run the anonymization
    local start_time=$(date +%s)

    # Build the command based on mode
    local cmd="python -m auto_anonymize \
        --input_dir \"$input_dir\" \
        --output_dir \"$output_dir\" \
        --headers_dir \"$headers_dir\" \
        --model_path \"$model_path\" \
        --device cpu"

    if [ "$mode" = "full" ]; then
        # Full mode with optional overview
        if [ -n "$overview_dir" ]; then
            cmd="$cmd --overview_dir \"$overview_dir\""
        fi
    elif [ "$mode" = "phi-only" ]; then
        # PHI-only mode
        cmd="$cmd --phi_only_mode --remove_phi_from_image"
    else
        log_error "Invalid mode: $mode"
        return 1
    fi

    # Execute the command
    if cd "$script_dir" && eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local mem_after=$(get_memory_usage)

        log_success "Completed: $relative_path (${duration}s, Memory: ${mem_before}MB → ${mem_after}MB)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Failed: $relative_path (${duration}s)"
        return 1
    fi
}

# Function to find and process all leaf directories
process_batch() {
    local input_root="$1"
    local output_root="$2"
    local headers_root="$3"
    local mode="$4"
    local overview_root="$5"  # May be empty
    local model_path="$6"
    local script_dir="$7"

    local total_processed=0
    local total_success=0
    local total_failed=0
    local failed_dirs=()

    log_info "Starting batch processing..."
    log_info "Input root: $input_root"
    log_info "Output root: $output_root"
    log_info "Headers root: $headers_root"
    log_info "Mode: $mode"
    if [ -n "$overview_root" ]; then
        log_info "Overview root: $overview_root"
    fi
    log_info "Model path: $model_path"
    log_info "Script directory: $script_dir"

    # Find all directories that contain DICOM files
    while IFS= read -r -d '' input_dir; do
        if has_dicom_files "$input_dir"; then
            # Calculate relative path from input root
            local relative_path="${input_dir#$input_root}"
            relative_path="${relative_path#/}"  # Remove leading slash

            # Create corresponding output paths
            local output_dir="$output_root/$relative_path"
            local headers_dir="$headers_root/$relative_path"
            local overview_dir=""

            if [ -n "$overview_root" ]; then
                overview_dir="$overview_root/$relative_path"
            fi

            total_processed=$((total_processed + 1))

            if process_directory "$input_dir" "$output_dir" "$headers_dir" "$relative_path" "$mode" "$overview_dir" "$model_path" "$script_dir"; then
                total_success=$((total_success + 1))
            else
                total_failed=$((total_failed + 1))
                failed_dirs+=("$relative_path")
            fi

            # Optional: Add a small delay between processing to help with memory management
            sleep 1
        fi
    done < <(find "$input_root" -type d -print0 | sort -z)

    # Print summary
    echo
    log_info "=== BATCH PROCESSING SUMMARY ==="
    log_info "Total directories processed: $total_processed"
    log_success "Successful: $total_success"

    if [ $total_failed -gt 0 ]; then
        log_error "Failed: $total_failed"
        log_error "Failed directories:"
        for dir in "${failed_dirs[@]}"; do
            log_error "  - $dir"
        done
    fi

    return $total_failed
}

# Function to validate inputs
validate_inputs() {
    local input_root="$1"
    local output_root="$2"
    local headers_root="$3"
    local mode="$4"
    local overview_root="$5"
    local model_path="$6"
    local script_dir="$7"

    # Check if input directory exists
    if [ ! -d "$input_root" ]; then
        log_error "Input directory does not exist: $input_root"
        exit 1
    fi

    # Validate mode
    if [ "$mode" != "full" ] && [ "$mode" != "phi-only" ]; then
        log_error "Invalid mode: $mode (must be 'full' or 'phi-only')"
        exit 1
    fi

    # Check if model file exists
    if [ ! -f "$model_path" ]; then
        log_error "Model file does not exist: $model_path"
        exit 1
    fi

    # Check if script directory exists
    if [ ! -d "$script_dir" ]; then
        log_error "Script directory does not exist: $script_dir"
        exit 1
    fi

    # Create output directories if they don't exist
    mkdir -p "$output_root"
    mkdir -p "$headers_root"

    # Create overview directory if specified
    if [ -n "$overview_root" ]; then
        mkdir -p "$overview_root"
    fi

    log_info "Validation passed"
}

# Function to preview what will be processed
preview_processing() {
    local input_root="$1"

    log_info "Preview of directories that will be processed:"
    local count=0

    while IFS= read -r -d '' input_dir; do
        if has_dicom_files "$input_dir"; then
            local relative_path="${input_dir#$input_root}"
            relative_path="${relative_path#/}"
            local dicom_count=$(count_dicom_files "$input_dir")

            count=$((count + 1))
            echo "  $count. $relative_path ($dicom_count DICOM files)"
        fi
    done < <(find "$input_root" -type d -print0 | sort -z)

    if [ $count -eq 0 ]; then
        log_warning "No directories with DICOM files found!"
        exit 1
    fi

    echo
    log_info "Total directories to process: $count"
}

# Main script execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Validate mode argument
    if [ "$MODE" != "full" ] && [ "$MODE" != "phi-only" ]; then
        log_error "Invalid mode: $MODE"
        show_usage
        exit 1
    fi

    # Warn if overview_root is provided with phi-only mode
    if [ "$MODE" = "phi-only" ] && [ -n "$OVERVIEW_ROOT" ]; then
        log_warning "Overview directory ignored in phi-only mode"
        OVERVIEW_ROOT=""
    fi

    # Convert to absolute paths
    INPUT_ROOT=$(realpath "$INPUT_ROOT")
    OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")
    HEADERS_ROOT=$(realpath "$HEADERS_ROOT")

    if [ -n "$OVERVIEW_ROOT" ]; then
        OVERVIEW_ROOT=$(realpath "$OVERVIEW_ROOT")
    fi

    MODEL_PATH=$(realpath "$MODEL_PATH")
    SCRIPT_DIR=$(realpath "$SCRIPT_DIR")

    log_info "=== DICOM BATCH ANONYMIZATION SCRIPT ==="

    # Validate inputs
    validate_inputs "$INPUT_ROOT" "$OUTPUT_ROOT" "$HEADERS_ROOT" "$MODE" "$OVERVIEW_ROOT" "$MODEL_PATH" "$SCRIPT_DIR"

    # Show preview
    preview_processing "$INPUT_ROOT"

    # Ask for confirmation
    echo
    read -p "Do you want to proceed with processing? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Processing cancelled by user"
        exit 0
    fi

    # Start processing
    local start_time=$(date +%s)

    if process_batch "$INPUT_ROOT" "$OUTPUT_ROOT" "$HEADERS_ROOT" "$MODE" "$OVERVIEW_ROOT" "$MODEL_PATH" "$SCRIPT_DIR"; then
        local end_time=$(date +%s)
        local total_duration=$((end_time - start_time))
        log_success "All processing completed successfully in ${total_duration}s"
        exit 0
    else
        local end_time=$(date +%s)
        local total_duration=$((end_time - start_time))
        log_error "Some directories failed to process (${total_duration}s total)"
        exit 1
    fi
}

# Run the main function with all arguments
main "$@"