#!/bin/bash

# Check if the path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <destination_path>"
    exit 1
fi

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
GENERATIED_DROPBOX_ROOT=$SCRIPTPATH"/../generated/Dropbox"
DROPBOX_ROOT="NUS_Dropbox:/Nikita Kazeev/Wyckoff Transformer data/"
PATH_IN_DROPBOX=$(realpath -s --relative-to="$GENERATIED_DROPBOX_ROOT" "$1")
rclone copy "$DROPBOX_ROOT$PATH_IN_DROPBOX" "$GENERATIED_DROPBOX_ROOT/$PATH_IN_DROPBOX"
