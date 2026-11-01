#!/bin/sh
# Compile simd_explicit.c with the library's flags, once per tier, and run it.
# Pin it to one fast core for numbers worth reading: taskset -c 2 sh run.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/wm_simd_explicit
cc=${CC:-cc}
flags="-O3 -std=c11 -ffp-contract=off -fno-math-errno -fno-trapping-math -fno-thread-jumps -Wall -Wextra"
case "$(uname -m)" in
    x86_64 | amd64)
        $cc $flags "$here/simd_explicit.c" -lm -o "$out" && "$out" "x86-64 baseline"
        $cc $flags -mavx2 "$here/simd_explicit.c" -lm -o "$out" && "$out" "x86-64 -mavx2"
        ;;
    aarch64 | arm64)
        $cc $flags "$here/simd_explicit.c" -lm -o "$out" && "$out" "arm64"
        ;;
    *)
        $cc $flags "$here/simd_explicit.c" -lm -o "$out" && "$out" "$(uname -m)"
        ;;
esac
