#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?Provide the dataset parent directory}"
[[ "$DATA_ROOT" == / ]] || DATA_ROOT="${DATA_ROOT%/}"
DATASET_REPO_ID="$(
    find -L "$DATA_ROOT" -mindepth 1 -type d -name data -print0 \
        | while IFS= read -r -d '' d; do
            root="$(dirname "$d")"
            if [[ -f "$root/meta/info.json" && -d "$root/videos" ]]; then
                if [[ "$root" == "$DATA_ROOT" ]]; then
                    repo=.
                else
                    repo="${root#"${DATA_ROOT%/}/"}"
                fi
                # The trainer splits repo IDs on whitespace.
                if [[ "$repo" == *[[:space:]]* ]]; then
                    printf 'Dataset relative path contains whitespace: %s\n' "$repo" >&2
                    exit 1
                fi
                printf '%s\n' "$repo"
            fi
        done | shuf | tr '\n' ' '
)"
if [[ -z "${DATASET_REPO_ID//[[:space:]]/}" ]]; then
    printf 'No LeRobot video datasets found below: %s\n' "$DATA_ROOT" >&2
    exit 1
fi
printf '%s\n' "$DATASET_REPO_ID"
