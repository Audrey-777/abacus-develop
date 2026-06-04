#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULT_DIR="${SCRIPT_DIR}/results"
BIN="${BUILD_DIR}/openmp_remainder_benchmark"
CSV="${RESULT_DIR}/openmp_remainder_benchmark.csv"
LOG="${RESULT_DIR}/openmp_remainder_benchmark.log"

NATOMS="${NATOMS:-2000000}"
LJ_NATOMS="${LJ_NATOMS:-50000}"
LJ_NEIGH="${LJ_NEIGH:-60}"
REPEAT="${REPEAT:-5}"
CXX="${CXX:-g++}"

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"

{
  echo "=== Build & Environment ==="
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Compiler: $(${CXX} --version | head -n 1)"
  echo "NATOMS=${NATOMS}"
  echo "LJ_NATOMS=${LJ_NATOMS}"
  echo "LJ_NEIGH=${LJ_NEIGH}"
  echo "REPEAT=${REPEAT}"
  echo "Build cmd: ${CXX} -O3 -std=c++17 -fopenmp"
  echo ""
  echo "=== CPU Info ==="
  lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Core|^Socket|^NUMA" || true
  echo ""
} > "${LOG}"

echo "Building..." | tee -a "${LOG}"
"${CXX}" -O3 -std=c++17 -fopenmp \
    "${SCRIPT_DIR}/openmp_remainder_benchmark.cpp" \
    -o "${BIN}" 2>&1 | tee -a "${LOG}"

echo "Build OK. Running benchmarks..." | tee -a "${LOG}"

: > "${CSV}"
for threads in 1 2 4 8 16; do
  echo ""
  echo "--- OMP_NUM_THREADS=${threads} ---" | tee -a "${LOG}"
  export OMP_NUM_THREADS="${threads}"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_PLACES="${OMP_PLACES:-cores}"
  tmp_csv="${RESULT_DIR}/run_${threads}.csv"
  "${BIN}" \
      --threads "${threads}" \
      --natoms "${NATOMS}" \
      --lj-natoms "${LJ_NATOMS}" \
      --lj-neigh "${LJ_NEIGH}" \
      --repeat "${REPEAT}" \
      > "${tmp_csv}"
  if [[ "${threads}" == "1" ]]; then
    cat "${tmp_csv}" >> "${CSV}"
  else
    tail -n +2 "${tmp_csv}" >> "${CSV}"
  fi
  echo "  done." | tee -a "${LOG}"
done

echo ""
echo "=== Results ===" | tee -a "${LOG}"
echo "CSV: ${CSV}" | tee -a "${LOG}"
echo "LOG: ${LOG}" | tee -a "${LOG}"

# Quick summary
echo ""
echo "=== Speedup Summary ===" | tee -a "${LOG}"
for kernel in rescale_vel msst_vel_sum msst_propagate_vel nhc_vel_baro fire_check_fire_mix fire_check_fire_zero lj_runner; do
  echo -n "${kernel}: " | tee -a "${LOG}"
  grep "^${kernel}," "${CSV}" | awk -F, '{printf "t=%s %.2fx  ", $2, $5}' | tee -a "${LOG}"
  echo "" | tee -a "${LOG}"
done
