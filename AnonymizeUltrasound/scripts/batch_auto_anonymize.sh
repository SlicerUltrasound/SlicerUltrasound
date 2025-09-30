#!/bin/bash

# batch_anonymize.sh
# Script to batch process nested DICOM directories using auto_anonymize.py
# Usage: ./batch_anonymize.sh <input_root_dir> <output_root_dir> <headers_root_dir>

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

    local dicom_count=$(count_dicom_files "$input_dir")
    local mem_before=$(get_memory_usage)

    log_info "Processing: $relative_path ($dicom_count DICOM files)"
    log_info "Memory usage before: ${mem_before}MB"

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$headers_dir"

    # Change to the AnonymizeUltrasound/scripts directory
    local script_dir="/Users/dqdinh/workspace/source/SlicerUltrasound/AnonymizeUltrasound/scripts"

    # Run the anonymization
    local start_time=$(date +%s)

    if cd "$script_dir" && python -m auto_anonymize \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --headers_dir "$headers_dir" \
        --model_path "/Users/dqdinh/workspace/source/SlicerUltrasound/AnonymizeUltrasound/Resources/checkpoints/baseline_unet_dsnt.pt" \
        --device cpu \
        --phi_only_mode \
        --remove_phi_from_image; then

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

    local total_processed=0
    local total_success=0
    local total_failed=0
    local failed_dirs=()

    log_info "Starting batch processing..."
    log_info "Input root: $input_root"
    log_info "Output root: $output_root"
    log_info "Headers root: $headers_root"

    # Find all directories that contain DICOM files
    while IFS= read -r -d '' input_dir; do
        if has_dicom_files "$input_dir"; then
            # Calculate relative path from input root
            local relative_path="${input_dir#$input_root}"
            relative_path="${relative_path#/}"  # Remove leading slash

            # Create corresponding output paths
            local output_dir="$output_root/$relative_path"
            local headers_dir="$headers_root/$relative_path"

            total_processed=$((total_processed + 1))

            if process_directory "$input_dir" "$output_dir" "$headers_dir" "$relative_path"; then
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

    # Check if input directory exists
    if [ ! -d "$input_root" ]; then
        log_error "Input directory does not exist: $input_root"
        exit 1
    fi

    # Check if model file exists
    local model_path="/Users/dqdinh/workspace/source/SlicerUltrasound/AnonymizeUltrasound/Resources/checkpoints/baseline_unet_dsnt.pt"
    if [ ! -f "$model_path" ]; then
        log_error "Model file does not exist: $model_path"
        exit 1
    fi

    # Check if script directory exists
    local script_dir="/Users/dqdinh/workspace/source/SlicerUltrasound/AnonymizeUltrasound/scripts"
    if [ ! -d "$script_dir" ]; then
        log_error "Script directory does not exist: $script_dir"
        exit 1
    fi

    # Create output directories if they don't exist
    mkdir -p "$output_root"
    mkdir -p "$headers_root"

    log_info "Validation passed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <input_root_dir> <output_root_dir> <headers_root_dir>"
    echo
    echo "Example:"
    echo "  $0 /path/to/R21-Batch001-2Pts /path/to/output /path/to/headers"
    echo
    echo "This script will:"
    echo "  1. Find all directories containing DICOM files (.dcm, .dicom)"
    echo "  2. Process each directory with auto_anonymize.py"
    echo "  3. Preserve the directory structure in output locations"
    echo "  4. Use CPU processing to avoid memory issues"
    echo "  5. Apply PHI-only mode with image redaction"
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
    # Check arguments
    if [ $# -ne 3 ]; then
        show_usage
        exit 1
    fi

    local input_root="$1"
    local output_root="$2"
    local headers_root="$3"

    # Convert to absolute paths
    input_root=$(realpath "$input_root")
    output_root=$(realpath "$output_root")
    headers_root=$(realpath "$headers_root")

    log_info "=== DICOM BATCH ANONYMIZATION SCRIPT ==="

    # Validate inputs
    validate_inputs "$input_root" "$output_root" "$headers_root"

    # Show preview
    preview_processing "$input_root"

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

    if process_batch "$input_root" "$output_root" "$headers_root"; then
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