#!/usr/bin/env bash
# Make the golden files for the pysmsim tests from the Fortran oracle.
#
# Usage:
#   oracle/make_golden.sh /path/to/smsim.2023-03-03_version [single|double] [OUT_DIR]
#
# The second argument picks the build (default: double, the main reference
# for the Python port). Run oracle/build.sh first. The golden files are
# written to OUT_DIR (default: tests/golden/<build>/), replacing what is
# there; work happens in oracle/build/golden/<build>/. The distribution is
# never modified.
#
# For every shipped .params file this runs:
#   rv/       tmrsk_loop_rv_drvr with oracle/inputs/tmrsk_loop_rv_drvr.ctl
#   fas/      fmrsk_loop_fas_drvr with oracle/inputs/fmrsk_loop_fas_drvr.ctl
#   td/       tmrsk_loop_td_drvr with oracle/inputs/tmrsk_loop_td_drvr.ctl
#   factors/  factor_dump at M 4, 6, 8 and Rrup 5, 20, 100 km, on 41
#             frequencies from 0.01 to 100 Hz
# and saves each output as <dir>/<params stem>[.M<m>.R<rrup>].<ext>.
# The control files take k0 and stress from the params file; their headers
# list the magnitudes, distances, periods and frequencies.
#
# Set SKIP_TD=1 to leave out the time-domain runs, which take longest.

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 /path/to/smsim.2023-03-03_version [single|double] [OUT_DIR]" >&2
    exit 2
fi
PRECISION=${2:-double}
case "$PRECISION" in
    single|double) ;;
    *) echo "error: build must be 'single' or 'double'" >&2; exit 2 ;;
esac

SMSIM_DIR=$(cd "$1" && pwd)
ORACLE_DIR=$(cd "$(dirname "$0")" && pwd)
BIN_DIR="$ORACLE_DIR/build/bin/$PRECISION"
RUN_DIR="$ORACLE_DIR/build/golden/$PRECISION"
OUT_DIR=${3:-"$ORACLE_DIR/../tests/golden/$PRECISION"}
SKIP_TD=${SKIP_TD:-0}

for prog in tmrsk_loop_rv_drvr fmrsk_loop_fas_drvr tmrsk_loop_td_drvr factor_dump; do
    if [ ! -x "$BIN_DIR/$prog" ]; then
        echo "error: $BIN_DIR/$prog not found; run oracle/build.sh first" >&2
        exit 1
    fi
done

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
rm -rf "$OUT_DIR/rv" "$OUT_DIR/fas" "$OUT_DIR/td" "$OUT_DIR/factors"
mkdir -p "$OUT_DIR/rv" "$OUT_DIR/fas" "$OUT_DIR/factors"
[ "$SKIP_TD" = 1 ] || mkdir -p "$OUT_DIR/td"

# prepare <dir>: params files with bare .pars file names (they are written
# with Windows paths, \smsim\...), and the .pars files themselves
prepare() {
    mkdir -p "$1"
    for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
        tr -d '\r' < "$f" | sed 's|\\smsim\\||' > "$1/$(basename "$f")"
    done
    for f in "$SMSIM_DIR"/params_and_pars_files/*.pars; do
        tr -d '\r' < "$f" > "$1/$(basename "$f")"
    done
}

# run_driver <program> <out subdir> <params file>
run_driver() {
    prog=$1
    sub=$2
    params=$3
    stem=${params%.params}
    dir="$RUN_DIR/$sub/$stem"
    prepare "$dir"
    sed -e "s|@PARAMS@|$params|" -e "s|@COL@|$prog.col|" \
        "$ORACLE_DIR/inputs/$prog.ctl" > "$dir/$prog.ctl"
    echo "$sub: $stem"
    (cd "$dir" && echo "$prog.ctl" | "$BIN_DIR/$prog" > stdout.txt 2>&1)
    cp "$dir/$prog.col" "$OUT_DIR/$sub/$stem.col"
}

for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    params=$(basename "$f")
    run_driver tmrsk_loop_rv_drvr rv "$params"
    run_driver fmrsk_loop_fas_drvr fas "$params"
    [ "$SKIP_TD" = 1 ] || run_driver tmrsk_loop_td_drvr td "$params"
done

dir="$RUN_DIR/factors"
prepare "$dir"
for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    params=$(basename "$f")
    stem=${params%.params}
    echo "factors: $stem"
    for m in 4.0 6.0 8.0; do
        for rrup in 5.0 20.0 100.0; do
            out="$stem.M$m.R$rrup.txt"
            (cd "$dir" && "$BIN_DIR/factor_dump" "$params" "$m" "$rrup" "$out" 41 0.01 100.0 \
                > /dev/null)
            cp "$dir/$out" "$OUT_DIR/factors/$out"
        done
    done
done

cp "$BIN_DIR/BUILD_INFO.txt" "$OUT_DIR/BUILD_INFO.txt"
echo "Golden files written to $OUT_DIR"
