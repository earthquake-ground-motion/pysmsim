#!/usr/bin/env bash
# Run the shipped tmrsk_loop_rv_drvr example with the oracle build and compare
# the result with the reference .col file from the SMSIM distribution.
#
# Usage:
#   oracle/run_example.sh /path/to/smsim.2023-03-03_version [single|double]
#
# The second argument picks the build (default: single). Run oracle/build.sh
# first. Work happens in oracle/build/run_example/<build>/; the distribution
# is never modified.
#
# The example inputs need three adjustments for the 2023 program:
#   - the control file is dated 01/22/21, but the program expects 03/03/23;
#   - the 2023 control file has an extra "Mech for Case 2" line (1 = strike-slip;
#     CY14 Ztor is the same for strike-slip and normal faults);
#   - the params file names the .pars files with Windows paths (\smsim\...),
#     which are reduced to bare file names.
#
# Environment variables:
#   TOL  maximum allowed relative difference in the ground-motion columns
#        (default: 2e-3)

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 /path/to/smsim.2023-03-03_version [single|double]" >&2
    exit 2
fi
PRECISION=${2:-single}
case "$PRECISION" in
    single|double) ;;
    *) echo "error: build must be 'single' or 'double'" >&2; exit 2 ;;
esac

SMSIM_DIR=$(cd "$1" && pwd)
ORACLE_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$ORACLE_DIR/build/bin/$PRECISION/tmrsk_loop_rv_drvr"
RUN_DIR="$ORACLE_DIR/build/run_example/$PRECISION"
EXAMPLE="$SMSIM_DIR/example.tmrsk_loop_rv_drvr"
PARAMS="wna.scf.bt15acr_fff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms.params"
TOL=${TOL:-2e-3}

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found; run oracle/build.sh first" >&2
    exit 1
fi

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Control file: new revision date, and the mechanism line after the
# h4rjb2rrup case value
tr -d '\r' < "$EXAMPLE/tmrsk_loop_rv_drvr.ctl" | awk '
    NR == 3 { print "   03/03/23"; next }
    { print }
    /^! Case for h4rjb2rrup:/ {
        getline; print
        print "! Mech for Case 2 (placeholder needed if case 2 /= 0): 1, 2, 3 for SS, NS, RS (CY14 Ztor is the same for NS, SS)"
        print " 1"
    }' > tmrsk_loop_rv_drvr.ctl

# Params file with bare .pars file names, and the .pars files themselves
tr -d '\r' < "$EXAMPLE/$PARAMS" | sed 's|\\smsim\\||' > "$PARAMS"
for f in "$SMSIM_DIR"/params_and_pars_files/*.pars; do
    tr -d '\r' < "$f" > "$(basename "$f")"
done

tr -d '\r' < "$EXAMPLE/tmrsk_loop_rv_drvr.col" > reference.col

echo "Running tmrsk_loop_rv_drvr ($PRECISION build) in $RUN_DIR"
echo "tmrsk_loop_rv_drvr.ctl" | "$BIN" > stdout.txt 2>&1

# Compare the numeric rows. Columns 1-12 (scenario values) must match to the
# printed digits, apart from T and F (columns 2-3), which are printed with 6
# significant digits. Columns 13 onwards are ground motions and are compared
# with the relative tolerance TOL.
awk -v tol="$TOL" '
    function isnum(s) { return s ~ /^[-+]?[0-9.]+([eE][-+]?[0-9]+)?$/ }
    function rel(a, b) { if (a == 0) return (b == 0) ? 0 : 1; return (a - b) / a }
    FNR == NR { if (isnum($1)) ref[++nr] = $0; next }
    isnum($1) {
        n++
        if (n > nr) { extra++; next }
        split(ref[n], r); m = split($0, x)
        for (j = 1; j <= m; j++) {
            d = rel(r[j] + 0, x[j] + 0)
            if (d < 0) d = -d
            if (j >= 13) { if (d > maxgm) maxgm = d; print d > "gm_reldiff.txt" }
            else if (j >= 2 && j <= 3) { if (d > 1e-5) bad++ }
            else if (r[j] != x[j]) bad++
        }
    }
    END {
        printf "rows: reference %d, oracle %d\n", nr, n
        printf "scenario columns that differ: %d\n", bad + 0
        printf "largest relative difference in ground motions: %.2e (tolerance %s)\n", maxgm, tol
        if (n != nr || extra || bad || maxgm > tol + 0) { print "RESULT: FAIL"; exit 1 }
        print "RESULT: PASS"
    }' reference.col tmrsk_loop_rv_drvr.col

sort -g gm_reldiff.txt | awk '{ v[NR] = $1 } END {
    printf "ground-motion relative differences: median %.1e, 90th pct %.1e, 99th pct %.1e\n",
        v[int(NR * 0.5) + 1], v[int(NR * 0.9) + 1], v[int(NR * 0.99) + 1] }'
