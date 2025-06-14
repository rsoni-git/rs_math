#!/usr/bin/env bash

set -e

_CURR_DIR=$(cd "$(dirname "${0}")" && pwd)
cd "${_CURR_DIR}"

[[ -s '100x100x3.png' ]] || magick -size 100x100 -depth 8 -define png:color-type=2 canvas:white 100x100x3.png
[[ -s '100x3x3.png' ]] || magick -size 100x3 -depth 8 -define png:color-type=2 canvas:white 100x3x3.png

[[ -s '1000x1000x3.png' ]] || magick -size 1000x1000 -depth 8 -define png:color-type=2 canvas:white 1000x1000x3.png
[[ -s '1000x3x3.png' ]] || magick -size 1000x3 -depth 8 -define png:color-type=2 canvas:white 1000x3x3.png

[[ -s '10000x10000x3.png' ]] || magick -size 10000x10000 -depth 8 -define png:color-type=2 canvas:white 10000x10000x3.png
[[ -s '10000x1000x3.png' ]] || magick -size 10000x3 -depth 8 -define png:color-type=2 canvas:white 10000x3x3.png
