#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
review_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${review_dir}/build"
reports_dir="${review_dir}/reports"

mkdir -p "${build_dir}" "${reports_dir}"

compiler="${CXX:-g++}"
binary="${build_dir}/parallel_optimize_benchmark"
csv="${reports_dir}/parallel_optimize_benchmark.csv"
log="${reports_dir}/parallel_optimize_benchmark.log"

nat="${NAT:-2000000}"
repeats="${REPEATS:-5}"
threads="${THREADS:-1,2,4,8}"
omp_proc_bind="${OMP_PROC_BIND:-close}"
omp_places="${OMP_PLACES:-cores}"

"${compiler}" -O3 -std=c++17 -fopenmp \
    "${script_dir}/parallel_optimize_benchmark.cpp" \
    -o "${binary}"

{
    echo "binary=${binary}"
    echo "compiler=${compiler}"
    echo "nat=${nat}"
    echo "repeats=${repeats}"
    echo "threads=${threads}"
    echo "OMP_PROC_BIND=${omp_proc_bind}"
    echo "OMP_PLACES=${omp_places}"
} > "${log}"

OMP_PROC_BIND="${omp_proc_bind}" \
OMP_PLACES="${omp_places}" \
"${binary}" --nat "${nat}" --repeats "${repeats}" --threads "${threads}" \
    > "${csv}"

echo "wrote ${csv}"
echo "wrote ${log}"
