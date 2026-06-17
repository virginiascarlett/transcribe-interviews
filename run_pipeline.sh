#!/usr/bin/env bash

# Runs the full interview transcription pipeline, skipping steps already done.

set -euo pipefail

# ── Load environment variables ────────────────────────────────────────────────

if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found. Please create it before running." >&2
    exit 1
fi

# Export variables from .env, ignoring comments and blank lines
set -o allexport
# shellcheck disable=SC1091
source .env
set +o allexport

# ── Validate required variables ───────────────────────────────────────────────

for var in DATA_DIR DATA_SUBDIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set in .env" >&2
        exit 1
    fi
done

SUBDIR="${DATA_DIR}/${DATA_SUBDIR}"

if [[ ! -d "$SUBDIR" ]]; then
    echo "ERROR: Data subdirectory '$SUBDIR' does not exist." >&2
    exit 1
fi

# ── Helper functions ──────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

skip() {
    echo "[$(date '+%H:%M:%S')] SKIP: $*"
}

# Returns the count of files matching a glob, or 0 if none exist.
count_files() {
    # Use find to avoid glob expansion issues when no files match.
    # If a non-zero exit code is returned (e.g., if an intermediate
    # dir/file doesn't exist yet), default to 0.
    # silence "No such file or directory" warnings because we are
    # creating the files as we go.
    find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' ' || echo 0
}

# ── Step 1: Rename recording to recording.mp4 ─────────────────────────────────

log "=== Step 1: Locate and rename recording ==="

RECORDING="${SUBDIR}/recording.mp4"

if [[ -f "$RECORDING" ]]; then
    skip "recording.mp4 already exists."
else
    # Find any .mp4 file in the subdirectory that isn't already recording.mp4
    candidates=()
    for f in "$SUBDIR"/*.mp4; do
        # skip to the next file if the condition is true
        # basename strips the directory, leaving just the filename
        [[ "$(basename "$f")" == "recording.mp4" ]] && continue
        [[ "$(basename "$f")" == chunk*.mp4 ]] && continue
        candidates+=("$f")
    done
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "ERROR: No source .mp4 file found in '$SUBDIR' to rename." >&2
        exit 1
    elif [[ ${#candidates[@]} -gt 1 ]]; then
        echo "ERROR: Multiple .mp4 files found in '$SUBDIR'. Please ensure only one source recording is present:" >&2
        printf '  %s\n' "${candidates[@]}" >&2
        exit 1
    fi

    log "Renaming '${candidates[0]}' to 'recording.mp4'..."
    mv "${candidates[0]}" "$RECORDING"
    log "Renamed successfully."
fi

# ── Step 2: Split recording into chunks ───────────────────────────────────────

log "=== Step 2: Split recording into chunks ==="

chunk_count=$(count_files "$SUBDIR" "chunk*.mp4")

if [[ "$chunk_count" -gt 0 ]]; then
    skip "Found $chunk_count chunk .mp4 file(s). Skipping split."
else
    log "Running split_recording.sh..."
    ./split_recording.sh
    log "Splitting complete."
fi

# Recount chunks — needed by later steps to verify completeness
chunk_count=$(count_files "$SUBDIR" "chunk*.mp4")

if [[ "$chunk_count" -eq 0 ]]; then
    echo "ERROR: No chunk .mp4 files found after splitting." >&2
    exit 1
fi

log "Chunk count: $chunk_count"

# ── Step 3a: Transcribe chunks ────────────────────────────────────────────────

log "=== Step 3a: Transcribe chunks ==="

transcript_count=$(count_files "${SUBDIR}/transcripts" "transcript*.txt")

if [[ "$transcript_count" -ge "$chunk_count" ]]; then
    skip "Found $transcript_count transcript(s) for $chunk_count chunk(s). Skipping transcription."
else
    log "Found $transcript_count transcript(s) but $chunk_count chunk(s). Running transcribe.py..."
    ./transcribe.py
    log "Transcription complete."
fi

# ── Step 3b: Convert chunks to WAV ───────────────────────────────────────────

log "=== Step 3b: Convert chunks to WAV ==="

wav_count=$(count_files "$SUBDIR" "chunk*.wav")

if [[ "$wav_count" -ge "$chunk_count" ]]; then
    skip "Found $wav_count .wav file(s) for $chunk_count chunk(s). Skipping conversion."
else
    log "Found $wav_count .wav file(s) but $chunk_count chunk(s). Running convert_to_wav.sh..."
    ./convert_to_wav.sh
    log "WAV conversion complete."
fi

# ── Step 3c: Diarize chunks ───────────────────────────────────────────────────

log "=== Step 3c: Diarize chunks ==="

diarization_count=$(count_files "${SUBDIR}/diarizations" "diarization*.txt")

if [[ "$diarization_count" -ge "$chunk_count" ]]; then
    skip "Found $diarization_count diarization(s) for $chunk_count chunk(s). Skipping diarization."
else
    log "Found $diarization_count diarization(s) but $chunk_count chunk(s). Running diarize.py..."
    ./diarize.py
    log "Diarization complete."
fi

# ── Step 4: Merge transcripts and diarizations ────────────────────────────────

log "=== Step 4: Merge transcripts and diarizations ==="

merged_count=$(count_files "${SUBDIR}/diarized_transcripts_raw" "merged*.txt")

if [[ "$merged_count" -ge "$chunk_count" ]]; then
    skip "Found $merged_count merged file(s) for $chunk_count chunk(s). Skipping merge."
else
    log "Found $merged_count merged file(s) but $chunk_count chunk(s). Running merge.py..."
    ./merge.py
    log "Merge complete."
fi

# ── Step 5: Clean up merged transcripts ──────────────────────────────────────

log "=== Step 5: Clean up (cleanup.py) ==="

clean_count=$(count_files "${SUBDIR}/diarized_transcripts_clean" "chunk*.txt")

if [[ "$clean_count" -ge "$chunk_count" ]]; then
    skip "Found $clean_count cleaned file(s) for $chunk_count chunk(s). Skipping cleanup."
else
    log "Found $clean_count cleaned file(s) but $chunk_count chunk(s). Running cleanup.py..."
    ./cleanup.py
    log "Cleanup complete."
fi

# ── Step 6: Final cleanup ─────────────────────────────────────────────────────

log "=== Step 6: Final cleanup (final_cleanup.py) ==="

final_count=$(count_files "${SUBDIR}/diarized_transcripts_clean_final" "final*.txt")

if [[ "$final_count" -ge "$chunk_count" ]]; then
    skip "Found $final_count final file(s) for $chunk_count chunk(s). Skipping final cleanup."
else
    log "Found $final_count final file(s) but $chunk_count chunk(s). Running final_cleanup.py..."
    ./final_cleanup.py
    log "Final cleanup complete."
fi

# ── Done ──────────────────────────────────────────────────────────────────────

log "=== Pipeline complete! ==="
log "Final transcripts are in: ${SUBDIR}/diarized_transcripts_clean_final/"