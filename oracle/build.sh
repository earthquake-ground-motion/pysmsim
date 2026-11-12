#!/usr/bin/env bash
# Build the SMSIM Fortran reference programs ("the oracle") with gfortran.
#
# Usage:
#   oracle/build.sh /path/to/smsim.2023-03-03_version
#
# The SMSIM distribution is never modified. Two builds are made, each from a
# fresh copy of the source converted from CRLF to LF line endings:
#
#   single  the code as written (mostly single precision), patched with
#           oracle/patches/*.patch; binaries in oracle/build/bin/single/
#   double  the same, plus oracle/patches-double/*.patch, compiled with
#           -fdefault-real-8 -fdefault-double-8 so that every REAL is 8 bytes;
#           binaries in oracle/build/bin/double/
#
# Each patch explains why it is needed in its header. The test-only programs
# in oracle/programs/ (written for pysmsim, not part of SMSIM) are copied in
# beside the source and built in both precisions too.
#
# Environment variables:
#   FC      Fortran compiler (default: gfortran)
#   FFLAGS  compiler flags for both builds (default: -O2)

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/smsim.2023-03-03_version" >&2
    exit 2
fi

SMSIM_DIR=$(cd "$1" && pwd)
ORACLE_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$ORACLE_DIR/build"
FC=${FC:-gfortran}
FFLAGS=${FFLAGS:--O2}

# SMSIM programs built for the golden tests
# (the source of FAS_DRVR is FAS_DRVR.FOR; the others end in .for)
PROGRAMS="tmrsk_loop_rv_drvr fmrsk_loop_fas_drvr tmrsk_loop_td_drvr gm_rv_drvr gm_td_drvr FAS_DRVR"
# Test-only programs from oracle/programs/
TEST_PROGRAMS="factor_dump response_dump td_dump"

if [ ! -d "$SMSIM_DIR/source_code" ]; then
    echo "error: $SMSIM_DIR/source_code not found" >&2
    exit 1
fi
if ! command -v "$FC" >/dev/null 2>&1; then
    echo "error: Fortran compiler '$FC' not found (set FC to override)" >&2
    exit 1
fi

# -Wno-align-commons: the promoted COMMON blocks need padding. That is safe
# here because every routine declares them through the same include files.
DOUBLE_FLAGS="-fdefault-real-8 -fdefault-double-8 -Wno-align-commons"

# build <name> <extra flags> <patch folder>...
build() {
    name=$1
    extra_flags=$2
    shift 2
    src_dir="$BUILD_DIR/src/$name"
    bin_dir="$BUILD_DIR/bin/$name"

    echo "== $name build"
    rm -rf "$src_dir" "$bin_dir"
    mkdir -p "$src_dir" "$bin_dir"
    for f in "$SMSIM_DIR"/source_code/*; do
        tr -d '\r' < "$f" > "$src_dir/$(basename "$f")"
    done

    applied=""
    for patch_dir in "$@"; do
        for p in "$ORACLE_DIR/$patch_dir"/*.patch; do
            [ -e "$p" ] || continue
            echo "Applying $patch_dir/$(basename "$p")"
            patch --quiet -d "$src_dir" -p1 < "$p"
            applied="$applied $patch_dir/$(basename "$p")"
        done
    done
    for prog in $TEST_PROGRAMS; do
        cp "$ORACLE_DIR/programs/$prog.for" "$src_dir/"
    done

    # The drivers pull in their subroutines with include statements, so
    # each program is a single compilation unit.
    for prog in $PROGRAMS $TEST_PROGRAMS; do
        echo "Compiling $prog"
        src="$prog.for"
        [ -e "$src_dir/$src" ] || src="$prog.FOR"
        (cd "$src_dir" && "$FC" $FFLAGS $extra_flags -o "$bin_dir/$prog" "$src")
    done

    # Record how the binaries were built, for the golden files
    {
        echo "build: $name"
        echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "smsim_dir: $SMSIM_DIR"
        echo "compiler: $("$FC" --version | head -n 1)"
        echo "fflags: $FFLAGS $extra_flags"
        echo "platform: $(uname -s) $(uname -m)"
        echo "patches:$applied"
        echo "test programs: $TEST_PROGRAMS"
    } > "$bin_dir/BUILD_INFO.txt"
    cat "$bin_dir/BUILD_INFO.txt"
}

build single "" patches
build double "$DOUBLE_FLAGS" patches patches-double

echo "Done. Binaries in $BUILD_DIR/bin/single and $BUILD_DIR/bin/double"
