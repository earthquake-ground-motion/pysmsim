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
#
# It also writes source/: factor_dump for each synthetic source case in
# oracle/inputs/source_cases.txt (ofr.params with the source lines replaced)
# at M 2, 3, 4.5, 5, 6 and 8, Rrup 20 km, on 21 frequencies from 0.01 to
# 100 Hz, saved as source/<case>.M<m>.txt.
# And path/: factor_dump for each synthetic path case in
# oracle/inputs/path_cases.txt (ofr.params with the finite-fault, spreading
# and Q lines replaced) at M 3, 5 and 7 and Rrup 0.5, 5, 30, 55 and 200 km,
# on 21 frequencies from 0.01 to 100 Hz, saved as
# path/<case>.M<m>.R<rrup>.txt.
# And site/: factor_dump for each synthetic site and filter case in
# oracle/inputs/site_cases.txt (ofr.params with the crustal-amplification,
# diminution, low-cut and high-cut lines replaced) at M 4, 6 and 8, Rrup
# 20 km, on 41 frequencies from 0.01 to 100 Hz, saved as
# site/<case>.M<m>.txt; and site_lines/: the site and filter value lines of
# each shipped .params file, which the tests read in place of the file.
# And response/: response_dump (spect_amp with each ground-motion and
# instrument response) for every shipped .params file at M 5 and 7, Rrup
# 10 km, on 41 frequencies from 0.01 to 100 Hz, with oscillators of 5%
# damping at 0.01, 0.1, 1 and 10 s, saved as response/<stem>.M<m>.txt.
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

for prog in tmrsk_loop_rv_drvr fmrsk_loop_fas_drvr tmrsk_loop_td_drvr factor_dump response_dump; do
    if [ ! -x "$BIN_DIR/$prog" ]; then
        echo "error: $BIN_DIR/$prog not found; run oracle/build.sh first" >&2
        exit 1
    fi
done

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
rm -rf "$OUT_DIR/rv" "$OUT_DIR/fas" "$OUT_DIR/td" "$OUT_DIR/factors" "$OUT_DIR/source" \
    "$OUT_DIR/path" "$OUT_DIR/site" "$OUT_DIR/site_lines" "$OUT_DIR/response"
mkdir -p "$OUT_DIR/rv" "$OUT_DIR/fas" "$OUT_DIR/factors" "$OUT_DIR/source" "$OUT_DIR/path" \
    "$OUT_DIR/site" "$OUT_DIR/site_lines" "$OUT_DIR/response"
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

dir="$RUN_DIR/response"
prepare "$dir"
for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    params=$(basename "$f")
    stem=${params%.params}
    echo "response: $stem"
    for m in 5.0 7.0; do
        out="$stem.M$m.txt"
        (cd "$dir" && "$BIN_DIR/response_dump" "$params" "$m" 10.0 "$out" 41 0.01 100.0 \
            0.05 0.01 0.1 1.0 10.0 > /dev/null)
        cp "$dir/$out" "$OUT_DIR/response/$out"
    done
done

# Synthetic source cases: replace the 4th, 5th and 6th value lines (not
# comments) of ofr.params, which are the shape, stress and scaling lines
dir="$RUN_DIR/source"
prepare "$dir"
grep -v '^ *#' "$ORACLE_DIR/inputs/source_cases.txt" | grep -v '^ *$' |
while IFS='|' read -r name shape stress scaling; do
    name=$(echo "$name" | tr -d ' ')
    echo "source: $name"
    awk -v l4="$shape" -v l5="$stress" -v l6="$scaling" '
        /^ *!/ { print; next }
        { n++ }
        n == 4 { print " " l4; next }
        n == 5 { print " " l5; next }
        n == 6 { print " " l6; next }
        { print }' "$dir/ofr.params" > "$dir/$name.params"
    for m in 2.0 3.0 4.5 5.0 6.0 8.0; do
        out="$name.M$m.txt"
        (cd "$dir" && "$BIN_DIR/factor_dump" "$name.params" "$m" 20.0 "$out" 21 0.01 100.0 \
            > /dev/null)
        cp "$dir/$out" "$OUT_DIR/source/$out"
    done
done

# Synthetic path cases: replace the 7th value line (finite fault), the
# spreading lines (8th to 14th) and the 15th (Q) of ofr.params. The comment
# lines among the spreading lines are replaced by one "!" line before the
# frequency-dependent parameters and one before r_ref: SMSIM's skipcmnt
# always skips the line before those reads, comment or not.
dir="$RUN_DIR/path"
prepare "$dir"
grep -v '^ *#' "$ORACLE_DIR/inputs/path_cases.txt" | grep -v '^ *$' |
while IFS='|' read -r name ff spreading q; do
    name=$(echo "$name" | tr -d ' ')
    echo "path: $name"
    awk -v ff="$ff" -v spreading="$spreading" -v q="$q" '
        /^ *!/ { if (n < 8 || n >= 14) print; next }
        { n++ }
        n == 7 { print " " ff; next }
        n == 8 {
            k = split(spreading, lines, ";")
            for (i = 1; i <= k; i++) {
                if (i == 2 || i == 3) print "!"
                print " " lines[i]
            }
            next
        }
        n > 8 && n < 15 { next }
        n == 15 { print " " q; next }
        { print }' "$dir/ofr.params" > "$dir/$name.params"
    for m in 3.0 5.0 7.0; do
        for rrup in 0.5 5.0 30.0 55.0 200.0; do
            out="$name.M$m.R$rrup.txt"
            (cd "$dir" && "$BIN_DIR/factor_dump" "$name.params" "$m" "$rrup" "$out" 21 0.01 100.0 \
                > /dev/null)
            cp "$dir/$out" "$OUT_DIR/path/$out"
        done
    done
done

# Synthetic site and filter cases: replace the crustal-amplification table
# (the 23rd value line, namps, and the namps lines after it), the
# diminution line, the low-cut line and the high-cut line of ofr.params.
# The comment lines before them are kept, as skipcmnt needs them.
dir="$RUN_DIR/site"
prepare "$dir"
grep -v '^ *#' "$ORACLE_DIR/inputs/site_cases.txt" | grep -v '^ *$' |
while IFS='|' read -r name amps dimin lowcut highcut; do
    name=$(echo "$name" | tr -d ' ')
    echo "site: $name"
    awk -v amps="$amps" -v dimin="$dimin" -v lowcut="$lowcut" -v highcut="$highcut" '
        /^ *!/ { print; next }
        { n++ }
        n == 23 {
            k = split(amps, points, ";")
            print " " k
            for (i = 1; i <= k; i++) print " " points[i]
            namps = $1
            next
        }
        n > 23 && n <= 23 + namps { next }
        n == 24 + namps { print " " dimin; next }
        n == 25 + namps { print " " lowcut; next }
        n == 26 + namps { print " " highcut; next }
        { print }' "$dir/ofr.params" > "$dir/$name.params"
    for m in 4.0 6.0 8.0; do
        out="$name.M$m.txt"
        (cd "$dir" && "$BIN_DIR/factor_dump" "$name.params" "$m" 20.0 "$out" 41 0.01 100.0 \
            > /dev/null)
        cp "$dir/$out" "$OUT_DIR/site/$out"
    done
done

# Site and filter value lines of the shipped .params files: from the
# crustal-amplification comment to the RV comment, without comments
for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    stem=$(basename "$f" .params)
    tr -d '\r' < "$f" |
        awk '/^!CRUSTAL AMPLIFICATION/ { on = 1 } /^!RV PARAMS/ { on = 0 }
             on && !/^ *!/ { sub(/^ +/, ""); print }' > "$OUT_DIR/site_lines/$stem.txt"
done

cp "$BIN_DIR/BUILD_INFO.txt" "$OUT_DIR/BUILD_INFO.txt"
echo "Golden files written to $OUT_DIR"
