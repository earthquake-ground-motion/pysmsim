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
# And rv_cases/: tmrsk_loop_rv_drvr for each synthetic RV case in
# oracle/inputs/rv_cases.txt (ofr.params with the diminution and RV lines
# replaced, and the damping of the control file), saved as
# rv_cases/<case>.col. And example/: the shipped example .col file and the
# same run with this build (oracle/run_example.sh). And td_series/: td_dump
# (the noise and acceleration of one time-domain simulation) for each
# synthetic case in oracle/inputs/td_cases.txt (ofr.params with the
# low-cut, window and timing lines replaced) at M 5, Rrup 10 km and M 7,
# Rrup 50 km, and for ofr.params as shipped at M 6, Rrup 20 km, saved
# gzipped as td_series/<case>.M<m>.R<rrup>.txt.gz.
# The control files take k0 and stress from the params file; their headers
# list the magnitudes, distances, periods and frequencies.
#
# And scenario/: the single-scenario drivers, fed their answers on stdin,
# for every shipped .params file: gm_rv_drvr (rv/, the .col file and the
# summary) and FAS_DRVR (fas/, on 41 frequencies from 0.01 to 100 Hz and
# oscillators at 0.1, 0.3, 1 and 3 s) at M 4, Rrup 10 km; M 6, Rrup 20 km;
# and M 7.5, Rrup 100 km; and gm_td_drvr (td/, the .col file and the
# summary, with the params file's number of simulations) at M 6, Rrup
# 20 km, saving the series of simulation 1 for ofr.params (gzipped). The RV
# and TD runs use oscillators of 5% damping at 0.01, 0.03, 0.1, 0.3, 1, 3
# and 10 s. Dates and times are removed from the summaries, saved as
# <stem>.M<m>.R<rrup>.sum.
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

for prog in tmrsk_loop_rv_drvr fmrsk_loop_fas_drvr tmrsk_loop_td_drvr factor_dump response_dump \
    td_dump gm_rv_drvr gm_td_drvr FAS_DRVR; do
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
    "$OUT_DIR/path" "$OUT_DIR/site" "$OUT_DIR/site_lines" "$OUT_DIR/response" \
    "$OUT_DIR/rv_cases" "$OUT_DIR/example" "$OUT_DIR/td_series" "$OUT_DIR/scenario"
mkdir -p "$OUT_DIR/rv" "$OUT_DIR/fas" "$OUT_DIR/factors" "$OUT_DIR/source" "$OUT_DIR/path" \
    "$OUT_DIR/site" "$OUT_DIR/site_lines" "$OUT_DIR/response" "$OUT_DIR/rv_cases" \
    "$OUT_DIR/example" "$OUT_DIR/td_series" "$OUT_DIR/scenario/rv" "$OUT_DIR/scenario/fas"
[ "$SKIP_TD" = 1 ] || mkdir -p "$OUT_DIR/td" "$OUT_DIR/scenario/td"

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

# Synthetic random-vibration cases: replace the diminution line (the 29th
# value line) and the RV line (the 32nd) of ofr.params, and the damping in
# the control file
dir="$RUN_DIR/rv_cases"
prepare "$dir"
grep -v '^ *#' "$ORACLE_DIR/inputs/rv_cases.txt" | grep -v '^ *$' |
while IFS='|' read -r name dimin rvline damping; do
    name=$(echo "$name" | tr -d ' ')
    damping=$(echo "$damping" | tr -d ' ')
    echo "rv_cases: $name"
    awk -v dimin="$dimin" -v rvline="$rvline" '
        /^ *!/ { print; next }
        { n++ }
        n == 29 { print " " dimin; next }
        n == 32 { print " " rvline; next }
        { print }' "$dir/ofr.params" > "$dir/$name.params"
    sed -e "s|@PARAMS@|$name.params|" -e "s|@COL@|$name.col|" \
        "$ORACLE_DIR/inputs/tmrsk_loop_rv_drvr.ctl" |
        awk -v damping="$damping" '
            done_damping == 0 && prev ~ /^!FractionalDamping/ { print " " damping; done_damping = 1; prev = $0; next }
            { print; prev = $0 }' > "$dir/$name.ctl"
    (cd "$dir" && echo "$name.ctl" | "$BIN_DIR/tmrsk_loop_rv_drvr" > "$name.stdout.txt" 2>&1)
    cp "$dir/$name.col" "$OUT_DIR/rv_cases/$name.col"
done

# Synthetic time-domain cases: replace the low-cut line (the 30th value
# line), the window line (35th) and the timing line (36th) of ofr.params,
# then write one simulation of each with td_dump. gzip -n leaves the time
# out of the file, so that the files are reproducible.
dir="$RUN_DIR/td_series"
prepare "$dir"
grep -v '^ *#' "$ORACLE_DIR/inputs/td_cases.txt" | grep -v '^ *$' |
while IFS='|' read -r name lowcut window timing; do
    name=$(echo "$name" | tr -d ' ')
    echo "td_series: $name"
    awk -v lowcut="$lowcut" -v window="$window" -v timing="$timing" '
        /^ *!/ { print; next }
        { n++ }
        n == 30 { print " " lowcut; next }
        n == 35 { print " " window; next }
        n == 36 { print " " timing; next }
        { print }' "$dir/ofr.params" > "$dir/$name.params"
    for mr in "5.0 10.0" "7.0 50.0"; do
        set -- $mr
        out="$name.M$1.R$2.txt"
        (cd "$dir" && "$BIN_DIR/td_dump" "$name.params" "$1" "$2" "$out" > /dev/null)
        gzip -n -c "$dir/$out" > "$OUT_DIR/td_series/$out.gz"
    done
done
out="ofr.M6.0.R20.0.txt"
(cd "$dir" && "$BIN_DIR/td_dump" ofr.params 6.0 20.0 "$out" > /dev/null)
gzip -n -c "$dir/$out" > "$OUT_DIR/td_series/$out.gz"

# Single-scenario drivers. They ask their questions on stdin; the answers
# are given one per line, in the order the programs ask them.
SCENARIO_PERIODS="0.01 0.03 0.1 0.3 1.0 3.0 10.0"
# strip_times <summary file>: the summary without dates and times
strip_times() {
    grep -v -e '^ *Date:' -e '^ *Time Start:' -e '^ *Time Stop:' -e '^ *Elapsed time' "$1"
}
dir="$RUN_DIR/scenario"
prepare "$dir"
for f in "$SMSIM_DIR"/params_and_pars_files/*.params; do
    params=$(basename "$f")
    stem=${params%.params}
    echo "scenario: $stem"
    for mr in "4.0 10.0" "6.0 20.0" "7.5 100.0"; do
        set -- $mr
        out="$stem.M$1.R$2"
        {
            echo "pysmsim golden file"; echo "$params"; echo "$out.rv.sum"
            echo y; echo y; echo "0.05 7"; for p in $SCENARIO_PERIODS; do echo "$p"; done
            echo "$1"; echo "$2"; echo "$out.rv.col"; echo n
        } | (cd "$dir" && "$BIN_DIR/gm_rv_drvr" > "$out.rv.stdout.txt" 2>&1)
        cp "$dir/$out.rv.col" "$OUT_DIR/scenario/rv/$out.col"
        strip_times "$dir/$out.rv.sum" > "$OUT_DIR/scenario/rv/$out.sum"
        {
            echo "pysmsim golden file"; echo "$params"; echo "$out.fas.sum"
            echo n; echo "0.01 100.0 41"
            echo y; echo y; echo "0.05 4"; for p in 0.1 0.3 1.0 3.0; do echo "$p"; done
            echo "$2"; echo "$1"; echo "$out.fas.col"; echo n
        } | (cd "$dir" && "$BIN_DIR/FAS_DRVR" > "$out.fas.stdout.txt" 2>&1)
        cp "$dir/$out.fas.col" "$OUT_DIR/scenario/fas/$out.col"
    done
    [ "$SKIP_TD" = 1 ] && continue
    out="$stem.M6.0.R20.0"
    {
        echo "pysmsim golden file"; echo "$params"; echo "$out.td.sum"
        echo y; echo y; echo "0.05 7"; for p in $SCENARIO_PERIODS; do echo "$p"; done
        echo 6.0; echo 20.0; echo "$out.td.col"
        if [ "$stem" = ofr ]; then echo y; echo 1; echo "$out.avd.col"; else echo n; fi
        echo n
    } | (cd "$dir" && "$BIN_DIR/gm_td_drvr" > "$out.td.stdout.txt" 2>&1)
    cp "$dir/$out.td.col" "$OUT_DIR/scenario/td/$out.col"
    strip_times "$dir/$out.td.sum" > "$OUT_DIR/scenario/td/$out.sum"
    if [ "$stem" = ofr ]; then
        gzip -n -c "$dir/$out.avd.col" > "$OUT_DIR/scenario/td/$out.avd.col.gz"
    fi
done

# The shipped example of tmrsk_loop_rv_drvr: the reference .col file from
# the distribution and the same run with this build (oracle/run_example.sh)
"$ORACLE_DIR/run_example.sh" "$SMSIM_DIR" "$PRECISION" > /dev/null || true
tr -d '\r' < "$SMSIM_DIR/example.tmrsk_loop_rv_drvr/tmrsk_loop_rv_drvr.col" \
    > "$OUT_DIR/example/tmrsk_loop_rv_drvr.shipped.col"
cp "$ORACLE_DIR/build/run_example/$PRECISION/tmrsk_loop_rv_drvr.col" \
    "$OUT_DIR/example/tmrsk_loop_rv_drvr.$PRECISION.col"

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
