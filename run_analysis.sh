#!/bin/bash
source /home/dene/miniconda3/etc/profile.d/conda.sh
conda activate rp2

LOG_DIR="/home/dene/rp2/logs"
INCOMING_DIR="/mnt/c/Users/denev/Sync/Syncthing"
CSV_LOG="/home/dene/rp2/logs/already_processed.csv"
RUNTIME_LOG="$LOG_DIR/pipeline_run_$(date +'%Y-%m-%d_%H-%M-%S').log"

SCRIPTS=(
  "/home/dene/rp2/00_auto_analysis_syncthing.py"
  "/home/dene/rp2/0_init.py"
  "/home/dene/rp2/1_audio_processing.py"
  "/home/dene/rp2/2_vowel_segment_separation.py"
  "/home/dene/rp2/3_extract_features.py"
  "/home/dene/rp2/4_prepare_dataframes.py"
  "/home/dene/rp2/5_modeling_features.py"
  "/home/dene/rp2/6_feature_extraction_statistics.py"
)

## checking for new .m4a files
existing=$(awk -F, 'NR>1 {print $1}' "$CSV_LOG" 2>/dev/null | sort)
current=$(find "$INCOMING_DIR" -name '*.m4a' -exec basename {} \; | sort)

if [ -n "$diff" ]; then
  echo "===== New files detected at $(date) =====" >> "$RUNTIME_LOG"

  for script in "${SCRIPTS[@]}"; do
    echo "--- Running: $(basename "$script") ---" >> "$RUNTIME_LOG"
    /home/dene/miniconda3/envs/rp2/bin/python "$script" >> "$RUNTIME_LOG" 2>&1
    echo "--- Finished: $(basename "$script") ---" >> "$RUNTIME_LOG"
  done

  echo "All scripts completed at $(date)" >> "$RUNTIME_LOG"
else
  echo "No new files found at $(date)." >> "$RUNTIME_LOG"
fi