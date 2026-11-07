#!/bin/sh
# Build sm_hashes.c against every tier this machine has and print the hashes.
# Every line for a kernel must end in the same hash, on every machine.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/sm_hashes
cc=${CC:-cc}
flags="-O3 -std=c11 -ffp-contract=off -fno-math-errno -fno-trapping-math -fno-thread-jumps -funroll-loops -I$here/.."
build() { # tier, extra flags
    $cc $flags $2 -DSM_TIER="$1" -DSM_TABLE="sm_table_$1" \
        "$here/../sm_kernels.c" "$here/sm_hashes.c" -lm -o "$out" && "$out"
}
# With SURMENTATIONS_BUILD_MARCH=native, the tier built for this machine too.
native=${SURMENTATIONS_BUILD_MARCH:-}
case "$(uname -m)" in
    x86_64 | amd64)
        build sse2 ""
        build avx2 "-mavx2"
        [ "$native" = native ] && build native "-march=native -mtune=native -DSM_NATIVE"
        ;;
    aarch64 | arm64)
        build neon ""
        [ "$native" = native ] && build native "-mcpu=native -DSM_NATIVE"
        ;;
    *) build portable "" ;;
esac
true
