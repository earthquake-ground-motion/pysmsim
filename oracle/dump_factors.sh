#!/usr/bin/env bash
# Print the spectral factors (oracle/programs/factor_dump.for) for every
# shipped .params file on a small grid of magnitudes and distances.
#
# Usage:
#   oracle/dump_factors.sh /path/to/smsim.2023-03-03_version [single|double]
#
# The second argument picks the build (default: double, the main reference
# for the Python port). Run oracle/build.sh first. Output goes to
# oracle/build/factors/<build>/, one file per params file and scenario,
# named <params stem>.M<m>.R<rrup>.txt; the distribution is never modified.
#
# Each file is checked: the product of the factors must equal spect_amp
# (column "fds") to within TOL (relative), which shows that no factor of
# the spectrum is missing from the dump.
#
# Environment variables:
#   MAGS  magnitudes (default: "4.0 6.0 8.0")
#   RRUPS rupture distances in km (default: "5.0 20.0 100.0")
#   TOL   allowed relative difference between product and fds
#         (default: 1e-5 for the single build, 1e-12 for the double build)

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 /path/to/smsim.2023-03-03_version [single|double]" >&2
    exit 2
fi
PRECISION=${2:-double}
case "$PRECISION" in
    single) DEFAULT_TOL=1e-5 ;;
    double) DEFAULT_TOL=1e-12 ;;
    *) echo "error: build must be 'single' or 'double'" >&2; exit 2 ;;
esac

SMSIM_DIR=$(cd "$1" && pwd)
ORACLE_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$ORACLE_DIR/build/bin/$PRECISION/factor_dump"
RUN_DIR="$ORACLE_DIR/build/factors/$PRECISION"
MAGS=${MAGS:-"4.0 6.0 8.0"}
RRUPS=${RRUPS:-"5.0 20.0 100.0"}
TOL=${TOL:-$DEFAULT_TOL}

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found; run oracle/build.sh first" >&2
    exit 1
fi

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Params files with bare .pars file names (they are written with Windows
# paths, \smsim\...), and the .pars files themselves
for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    tr -d '\r' < "$f" | sed 's|\\smsim\\||' > "$(basename "$f")"
done
for f in "$SMSIM_DIR"/params_and_pars_files/*.pars; do
    tr -d '\r' < "$f" > "$(basename "$f")"
done

nfail=0
for params in *.params; do
    stem=${params%.params}
    for m in $MAGS; do
        for rrup in $RRUPS; do
            out="$stem.M$m.R$rrup.txt"
            "$BIN" "$params" "$m" "$rrup" "$out" > /dev/null
            # Largest relative difference between product (column 12) and
            # fds (column 13), over the rows where fds is not zero
            maxd=$(awk '!/^!/ && $1 != "f" {
                    if ($13 != 0) { d = ($12 - $13) / $13; if (d < 0) d = -d
                                    if (d > maxd) maxd = d } }
                    END { printf "%.2e", maxd }' "$out")
            status=ok
            if awk -v d="$maxd" -v t="$TOL" 'BEGIN { exit !(d > t) }'; then
                status=FAIL
                nfail=$((nfail + 1))
            fi
            printf "%-4s %s (product vs fds: %s)\n" "$status" "$out" "$maxd"
        done
    done
done

rm -f ./*.params ./*.pars
if [ "$nfail" -gt 0 ]; then
    echo "RESULT: FAIL ($nfail files exceed tolerance $TOL)"
    exit 1
fi
echo "RESULT: PASS (tolerance $TOL)"
