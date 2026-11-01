#!/bin/sh
# Build wm_hashes.c against every tier this machine has and print the hashes.
# Every line for a kernel must end in the same hash, on every machine.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/wm_hashes
cc=${CC:-cc}
flags="-O3 -std=c11 -ffp-contract=off -fno-math-errno -fno-trapping-math -fno-thread-jumps -funroll-loops -I$here/.."
build() { # tier, extra flags
    $cc $flags $2 -DWM_TIER="$1" -DWM_TABLE="wm_table_$1" \
        "$here/../wm_kernels.c" "$here/wm_hashes.c" -lm -o "$out" && "$out"
}
# With WEATHERMENTATIONS_BUILD_MARCH=native, the tier built for this machine too.
native=${WEATHERMENTATIONS_BUILD_MARCH:-}
case "$(uname -m)" in
    x86_64 | amd64)
        build sse2 ""
        build avx2 "-mavx2"
        [ "$native" = native ] && build native "-march=native -mtune=native -DWM_NATIVE"
        ;;
    aarch64 | arm64)
        build neon ""
        [ "$native" = native ] && build native "-mcpu=native -DWM_NATIVE"
        ;;
    *) build portable "" ;;
esac
true
