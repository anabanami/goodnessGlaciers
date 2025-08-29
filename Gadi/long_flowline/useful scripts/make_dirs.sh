#!/usr/bin/env bash
set -euo pipefail

ids=(
005 025 033 039 063 064 088 089 127 133 165 172 177 178 213 214 270 271 287 288 318 319 345 346 396 399 412 420 436 437 463 464 519 520 546 547 552 574 590 591 636 639 667 674 694 696 725 726 768 769 783 794 807 826 828 843

)

for id in "${ids[@]}"; do
  # force base-10 to avoid octal interpretation (e.g., "074")
  padded=$(printf "%03d" "$((10#$id))")
  dir="${padded}"
  if [[ -d "$dir" ]]; then
    echo "exists: $dir"
  else
    mkdir -p -- "$dir"
    echo "created: $dir"
  fi
done