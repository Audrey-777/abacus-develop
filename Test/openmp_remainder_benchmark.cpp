/**
 * @file openmp_remainder_benchmark.cpp
 * @brief Microbenchmark for the 6 newly OpenMP-parallelized kernels
 *        in refactor/md-openmp-remainder branch.
 *
 * Kernels under test:
 *   1. rescale_vel       — velocity rescaling factor apply
 *   2. msst_vel_sum      — norm2 reduction
 *   3. msst_propagate_vel — exp-based velocity propagation
 *   4. nhc_vel_baro      — NPT per-atom velocity scaling
 *   5. fire_check_fire   — triple reduction + velocity mixing + zero
 *   6. lj_runner         — N² neighbor pair computation
 *
 * Usage:
 *   g++ -O3 -std=c++17 -fopenmp openmp_remainder_benchmark.cpp -o bench
 *   OMP_PROC_BIND=close OMP_PLACES=cores ./bench --threads 8 --natoms 2000000 --repeat 5
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

// ---------------------------------------------------------------------------
// lightweight Vec3 / Matrix types (no external dependencies)
// ---------------------------------------------------------------------------
struct Vec3
{
    double x, y, z;
    double norm2() const { return x * x + y * y + z * z; }
};

// ---------------------------------------------------------------------------
// benchmark data structure
// ---------------------------------------------------------------------------
struct Data
{
    int natom = 2000000;
    double md_dt = 0.5;
    double sqrt_factor = 1.05; // rescale_vel
    double alpha = 0.15;       // FIRE
    double sumforce = 12.0;    // FIRE
    double normvel = 8.0;      // FIRE

    // MSST state
    double msst_vis = 2.0;
    double omega_sd = 0.003;
    double vsum = 5000.0;
    double ucell_omega = 1000.0;
    int sd = 0;

    // NHC state
    double v_omega[6] = {0.01, 0.02, 0.03, 0.004, 0.005, 0.006};
    double mtk_term = 0.001;

    // atom data
    std::vector<Vec3> vel;
    std::vector<Vec3> vel_backup; // for kernels that modify vel
    std::vector<Vec3> force;
    std::vector<double> mass;
    std::vector<std::array<int, 3>> ionmbl;

    // LJ data
    int avg_neigh = 60; // average neighbours per atom
    std::vector<int> neigh_count;
    std::vector<int> neigh_offset;        // prefix sum of neigh_count
    std::vector<int> neigh_type;          // it2 per neighbour pair
    std::vector<double> neigh_dx;         // dtau.x per neighbour pair
    std::vector<double> neigh_dy;         // dtau.y per neighbour pair
    std::vector<double> neigh_dz;         // dtau.z per neighbour pair
    std::vector<double> lj_c6_diag;      // per type-pair C6
    std::vector<double> lj_c12_diag;     // per type-pair C12
    std::vector<double> en_shift_diag;   // per type-pair energy shift
    std::vector<double> lj_force_x;      // output force buffer
    std::vector<double> lj_force_y;
    std::vector<double> lj_force_z;
    double lj_potential = 0.0;
    double lj_virial[9] = {};
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
volatile double sink = 0.0;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int parse_int_arg(int argc, char** argv, const std::string& name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == name) return std::atoi(argv[i + 1]);
    return fallback;
}

double rval(int i, double scale) { return scale * (1.0 + ((i * 17) % 97) / 97.0); }

Data make_data(int natom, int avg_neigh)
{
    Data d;
    d.natom = natom;
    d.avg_neigh = avg_neigh;
    d.vel.resize(natom);
    d.vel_backup.resize(natom);
    d.force.resize(natom);
    d.mass.resize(natom);
    d.ionmbl.resize(natom);
    for (int i = 0; i < natom; ++i)
    {
        d.vel[i] = {rval(i, 0.001), rval(i + 11, 0.0012), rval(i + 23, 0.0009)};
        d.force[i] = {rval(i + 3, 0.004), rval(i + 5, 0.003), rval(i + 7, 0.005)};
        d.mass[i] = 10.0 + static_cast<double>(i % 13);
        d.ionmbl[i] = {1, (i % 11) == 0 ? 0 : 1, (i % 17) == 0 ? 0 : 1};
    }
    d.vel_backup = d.vel;

    // LJ synthetic data
    d.neigh_count.resize(natom);
    int total_pairs = 0;
    for (int i = 0; i < natom; ++i)
    {
        int nc = avg_neigh + ((i * 7) % 21) - 10; // avg_neigh ± 10
        if (nc < 0) nc = 0;
        d.neigh_count[i] = nc;
        total_pairs += nc;
    }
    d.neigh_offset.resize(natom + 1);
    d.neigh_offset[0] = 0;
    for (int i = 0; i < natom; ++i)
        d.neigh_offset[i + 1] = d.neigh_offset[i] + d.neigh_count[i];

    d.neigh_type.resize(total_pairs);
    d.neigh_dx.resize(total_pairs);
    d.neigh_dy.resize(total_pairs);
    d.neigh_dz.resize(total_pairs);
    for (int p = 0; p < total_pairs; ++p)
    {
        d.neigh_type[p] = (p * 3) % 4;
        d.neigh_dx[p] = rval(p, 0.5);
        d.neigh_dy[p] = rval(p + 1, 0.6);
        d.neigh_dz[p] = rval(p + 2, 0.7);
    }

    d.lj_force_x.resize(natom);
    d.lj_force_y.resize(natom);
    d.lj_force_z.resize(natom);

    // C6/C12/en_shift per type pair (4 types)
    d.lj_c6_diag.resize(16);  // type*4+type2
    d.lj_c12_diag.resize(16);
    d.en_shift_diag.resize(16);
    for (int ti = 0; ti < 4; ++ti)
        for (int tj = 0; tj < 4; ++tj)
        {
            int idx = ti * 4 + tj;
            d.lj_c6_diag[idx] = rval(idx, 0.02);
            d.lj_c12_diag[idx] = rval(idx + 41, 0.0001);
            d.en_shift_diag[idx] = rval(idx + 83, 0.005);
        }

    return d;
}

template <typename F>
double time_repeated(int repeat, F&& f)
{
    const auto start = Clock::now();
    for (int r = 0; r < repeat; ++r) f();
    return elapsed_ms(start, Clock::now()) / repeat;
}

// ---------------------------------------------------------------------------
// measurement output
// ---------------------------------------------------------------------------
struct Measurement
{
    std::string kernel;
    int threads;
    double serial_ms;
    double omp_ms;
    double max_abs_diff;
    double checksum;
};

void print_csv_header()
{
    std::cout << "kernel,threads,serial_ms,omp_ms,speedup,efficiency,max_abs_diff,checksum\n";
}

void print_measurement(const Measurement& m)
{
    double speedup = m.omp_ms > 0.0 ? m.serial_ms / m.omp_ms : 0.0;
    double efficiency = m.threads > 0 ? speedup / m.threads : 0.0;
    std::cout << m.kernel << ',' << m.threads << ','
              << std::fixed << std::setprecision(6)
              << m.serial_ms << ',' << m.omp_ms << ',' << speedup << ',' << efficiency << ','
              << std::scientific << m.max_abs_diff << ',' << m.checksum << '\n';
}

double checksum_vec(const std::vector<Vec3>& a)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i += 4097) s += a[i].x + 0.5 * a[i].y + 0.25 * a[i].z;
    return s;
}

double checksum_dbl(const std::vector<double>& a)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i += 4097) s += a[i];
    return s;
}

double max_diff_vec(const std::vector<Vec3>& a, const std::vector<Vec3>& b)
{
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        d = std::max(d, std::abs(a[i].x - b[i].x));
        d = std::max(d, std::abs(a[i].y - b[i].y));
        d = std::max(d, std::abs(a[i].z - b[i].z));
    }
    return d;
}

double max_diff_dbl(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

// ===================================================================
// Kernel 1: rescale_vel
// ===================================================================
void rescale_vel_serial(const Data& d, std::vector<Vec3>& vel)
{
    for (int i = 0; i < d.natom; ++i)
        vel[i].x *= d.sqrt_factor, vel[i].y *= d.sqrt_factor, vel[i].z *= d.sqrt_factor;
}

void rescale_vel_omp(const Data& d, std::vector<Vec3>& vel)
{
    const int n = d.natom;
    const double f = d.sqrt_factor;
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int i = 0; i < n; ++i)
        vel[i].x *= f, vel[i].y *= f, vel[i].z *= f;
}

// ===================================================================
// Kernel 2: msst_vel_sum
// ===================================================================
double msst_vel_sum_serial(const Data& d)
{
    double s = 0.0;
    for (int i = 0; i < d.natom; ++i) s += d.vel[i].norm2();
    return s;
}

double msst_vel_sum_omp(const Data& d)
{
    double s = 0.0;
    const int n = d.natom;
#pragma omp parallel for reduction(+:s) schedule(static) if (n >= 256)
    for (int i = 0; i < n; ++i) s += d.vel[i].norm2();
    return s;
}

// ===================================================================
// Kernel 3: msst_propagate_vel
// ===================================================================
void msst_propagate_vel_serial(const Data& d, std::vector<Vec3>& vel)
{
    const double dthalf = 0.5 * d.md_dt;
    const double fac = d.msst_vis * d.omega_sd * d.omega_sd / (d.vsum * d.ucell_omega);
    for (int i = 0; i < d.natom; ++i)
    {
        Vec3 C = {d.force[i].x / d.mass[i], d.force[i].y / d.mass[i], d.force[i].z / d.mass[i]};
        Vec3 D = {fac / d.mass[i], fac / d.mass[i], fac / d.mass[i]};
        (&D.x)[d.sd] -= 2.0 * d.omega_sd / d.ucell_omega;

        double ck[3] = {C.x, C.y, C.z};
        double dk[3] = {D.x, D.y, D.z};
        double* vk[3] = {&vel[i].x, &vel[i].y, &vel[i].z};

        for (int k = 0; k < 3; ++k)
        {
            if (std::fabs(dthalf * dk[k]) > 1e-6)
            {
                double expd = std::exp(dthalf * dk[k]);
                *vk[k] = expd * (ck[k] + dk[k] * (*vk[k]) - ck[k] / expd) / dk[k];
            }
            else
            {
                *vk[k] += (ck[k] + dk[k] * (*vk[k])) * dthalf
                        + 0.5 * (dk[k] * dk[k] * (*vk[k]) + ck[k] * dk[k]) * dthalf * dthalf;
            }
        }
    }
}

void msst_propagate_vel_omp(const Data& d, std::vector<Vec3>& vel)
{
    const double dthalf = 0.5 * d.md_dt;
    const double fac = d.msst_vis * d.omega_sd * d.omega_sd / (d.vsum * d.ucell_omega);
    const int n = d.natom;
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int i = 0; i < n; ++i)
    {
        Vec3 C = {d.force[i].x / d.mass[i], d.force[i].y / d.mass[i], d.force[i].z / d.mass[i]};
        Vec3 D = {fac / d.mass[i], fac / d.mass[i], fac / d.mass[i]};
        (&D.x)[d.sd] -= 2.0 * d.omega_sd / d.ucell_omega;

        double ck[3] = {C.x, C.y, C.z};
        double dk[3] = {D.x, D.y, D.z};
        double* vk[3] = {&vel[i].x, &vel[i].y, &vel[i].z};

        for (int k = 0; k < 3; ++k)
        {
            if (std::fabs(dthalf * dk[k]) > 1e-6)
            {
                double expd = std::exp(dthalf * dk[k]);
                *vk[k] = expd * (ck[k] + dk[k] * (*vk[k]) - ck[k] / expd) / dk[k];
            }
            else
            {
                *vk[k] += (ck[k] + dk[k] * (*vk[k])) * dthalf
                        + 0.5 * (dk[k] * dk[k] * (*vk[k]) + ck[k] * dk[k]) * dthalf * dthalf;
            }
        }
    }
}

// ===================================================================
// Kernel 4: nhc_vel_baro
// ===================================================================
void nhc_vel_baro_serial(const Data& d, std::vector<Vec3>& vel)
{
    double f[3];
    for (int j = 0; j < 3; ++j)
        f[j] = std::exp(-(d.v_omega[j] + d.mtk_term) * d.md_dt / 4.0);

    for (int i = 0; i < d.natom; ++i)
    {
        vel[i].x *= f[0];
        vel[i].y *= f[1];
        vel[i].z *= f[2];

        if (d.ionmbl[i][0])
            vel[i].x -= (vel[i].y * d.v_omega[5] + vel[i].z * d.v_omega[4]) * d.md_dt / 2.0;
        if (d.ionmbl[i][1])
            vel[i].y -= vel[i].z * d.v_omega[3] * d.md_dt / 2.0;

        vel[i].x *= f[0];
        vel[i].y *= f[1];
        vel[i].z *= f[2];
    }
}

void nhc_vel_baro_omp(const Data& d, std::vector<Vec3>& vel)
{
    double f[3];
    for (int j = 0; j < 3; ++j)
        f[j] = std::exp(-(d.v_omega[j] + d.mtk_term) * d.md_dt / 4.0);

    const int n = d.natom;
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int i = 0; i < n; ++i)
    {
        vel[i].x *= f[0];
        vel[i].y *= f[1];
        vel[i].z *= f[2];

        if (d.ionmbl[i][0])
            vel[i].x -= (vel[i].y * d.v_omega[5] + vel[i].z * d.v_omega[4]) * d.md_dt / 2.0;
        if (d.ionmbl[i][1])
            vel[i].y -= vel[i].z * d.v_omega[3] * d.md_dt / 2.0;

        vel[i].x *= f[0];
        vel[i].y *= f[1];
        vel[i].z *= f[2];
    }
}

// ===================================================================
// Kernel 5: fire_check_fire
// ===================================================================
struct FireResult
{
    double P, sumforce, normvel;
};

FireResult fire_check_fire_serial(const Data& d, std::vector<Vec3>& vel, bool do_zero)
{
    double P = 0.0, sf = 0.0, nv = 0.0;
    for (int i = 0; i < d.natom; ++i)
    {
        P += vel[i].x * d.force[i].x + vel[i].y * d.force[i].y + vel[i].z * d.force[i].z;
        sf += d.force[i].norm2();
        nv += vel[i].norm2();
    }
    double sumforce = std::sqrt(sf), normvel = std::sqrt(nv);

    if (do_zero)
    {
        for (int i = 0; i < d.natom; ++i)
            vel[i] = {0, 0, 0};
    }
    else
    {
        for (int i = 0; i < d.natom; ++i)
        {
            vel[i].x = (1.0 - d.alpha) * vel[i].x + d.alpha * d.force[i].x / sumforce * normvel;
            vel[i].y = (1.0 - d.alpha) * vel[i].y + d.alpha * d.force[i].y / sumforce * normvel;
            vel[i].z = (1.0 - d.alpha) * vel[i].z + d.alpha * d.force[i].z / sumforce * normvel;
        }
    }
    return {P, sumforce, normvel};
}

FireResult fire_check_fire_omp(const Data& d, std::vector<Vec3>& vel, bool do_zero)
{
    double P = 0.0, sf = 0.0, nv = 0.0;
    const int n = d.natom;
#pragma omp parallel for reduction(+:P, sf, nv) schedule(static) if (n >= 256)
    for (int i = 0; i < n; ++i)
    {
        P += vel[i].x * d.force[i].x + vel[i].y * d.force[i].y + vel[i].z * d.force[i].z;
        sf += d.force[i].norm2();
        nv += vel[i].norm2();
    }
    double sumforce = std::sqrt(sf), normvel = std::sqrt(nv);

    if (do_zero)
    {
#pragma omp parallel for schedule(static) if (n >= 256)
        for (int i = 0; i < n; ++i) vel[i] = {0, 0, 0};
    }
    else
    {
        const double a = d.alpha;
        const double ratio = a * normvel / sumforce;
        const double om = 1.0 - a;
#pragma omp parallel for schedule(static) if (n >= 256)
        for (int i = 0; i < n; ++i)
        {
            vel[i].x = om * vel[i].x + ratio * d.force[i].x;
            vel[i].y = om * vel[i].y + ratio * d.force[i].y;
            vel[i].z = om * vel[i].z + ratio * d.force[i].z;
        }
    }
    return {P, sumforce, normvel};
}

FireResult fire_check_fire_mix_serial(const Data& d, std::vector<Vec3>& vel)
{
    return fire_check_fire_serial(d, vel, false);
}

FireResult fire_check_fire_mix_omp(const Data& d, std::vector<Vec3>& vel)
{
    return fire_check_fire_omp(d, vel, false);
}

FireResult fire_check_fire_zero_serial(const Data& d, std::vector<Vec3>& vel)
{
    return fire_check_fire_serial(d, vel, true);
}

FireResult fire_check_fire_zero_omp(const Data& d, std::vector<Vec3>& vel)
{
    return fire_check_fire_omp(d, vel, true);
}

// ===================================================================
// Kernel 6: lj_runner (N² neighbour pairs)
// ===================================================================
void lj_runner_serial(const Data& d,
                      std::vector<double>& fx,
                      std::vector<double>& fy,
                      std::vector<double>& fz,
                      double& pot,
                      double vir[9])
{
    std::fill(fx.begin(), fx.end(), 0.0);
    std::fill(fy.begin(), fy.end(), 0.0);
    std::fill(fz.begin(), fz.end(), 0.0);
    pot = 0.0;
    for (int j = 0; j < 9; ++j) vir[j] = 0.0;

    for (int iat = 0; iat < d.natom; ++iat)
    {
        int nb = d.neigh_count[iat];
        int off = d.neigh_offset[iat];
        for (int p = 0; p < nb; ++p)
        {
            int idx = off + p;
            double dx = d.neigh_dx[idx], dy = d.neigh_dy[idx], dz = d.neigh_dz[idx];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r4 = r2 * r2;
            double r6 = r2 * r4;
            int it2 = d.neigh_type[idx];
            int pair_idx = (iat % 4) * 4 + it2;

            double c6 = d.lj_c6_diag[pair_idx];
            double c12 = d.lj_c12_diag[pair_idx];
            double es = d.en_shift_diag[pair_idx];

            double e = c12 / (r6 * r6) - c6 / r6 - es;
            pot += e;

            double r8 = r4 * r4;
            double r14 = r8 * r4 * r2;
            double coff = 12.0 * c12 / r14 - 6.0 * c6 / r8;
            double fx_ij = dx * coff, fy_ij = dy * coff, fz_ij = dz * coff;

            fx[iat] += fx_ij;
            fy[iat] += fy_ij;
            fz[iat] += fz_ij;

            vir[0] += dx * fx_ij; vir[1] += dx * fy_ij; vir[2] += dx * fz_ij;
            vir[3] += dy * fx_ij; vir[4] += dy * fy_ij; vir[5] += dy * fz_ij;
            vir[6] += dz * fx_ij; vir[7] += dz * fy_ij; vir[8] += dz * fz_ij;
        }
    }
}

void lj_runner_omp(const Data& d,
                   std::vector<double>& fx,
                   std::vector<double>& fy,
                   std::vector<double>& fz,
                   double& pot,
                   double vir[9])
{
    std::fill(fx.begin(), fx.end(), 0.0);
    std::fill(fy.begin(), fy.end(), 0.0);
    std::fill(fz.begin(), fz.end(), 0.0);
    pot = 0.0;
    for (int j = 0; j < 9; ++j) vir[j] = 0.0;

    const int n = d.natom;
#pragma omp parallel if (n >= 256)
    {
        double vl[9] = {};
        double pl = 0.0;

#pragma omp for schedule(dynamic, 32)
        for (int iat = 0; iat < n; ++iat)
        {
            int nb = d.neigh_count[iat];
            int off = d.neigh_offset[iat];
            for (int p = 0; p < nb; ++p)
            {
                int idx = off + p;
                double dx = d.neigh_dx[idx], dy = d.neigh_dy[idx], dz = d.neigh_dz[idx];
                double r2 = dx * dx + dy * dy + dz * dz;
                double r4 = r2 * r2;
                double r6 = r2 * r4;
                int it2 = d.neigh_type[idx];
                int pair_idx = (iat % 4) * 4 + it2;

                double c6 = d.lj_c6_diag[pair_idx];
                double c12 = d.lj_c12_diag[pair_idx];
                double es = d.en_shift_diag[pair_idx];

                double e = c12 / (r6 * r6) - c6 / r6 - es;
                pl += e;

                double r8 = r4 * r4;
                double r14 = r8 * r4 * r2;
                double coff = 12.0 * c12 / r14 - 6.0 * c6 / r8;
                double fx_ij = dx * coff, fy_ij = dy * coff, fz_ij = dz * coff;

                fx[iat] += fx_ij;
                fy[iat] += fy_ij;
                fz[iat] += fz_ij;

                vl[0] += dx * fx_ij; vl[1] += dx * fy_ij; vl[2] += dx * fz_ij;
                vl[3] += dy * fx_ij; vl[4] += dy * fy_ij; vl[5] += dy * fz_ij;
                vl[6] += dz * fx_ij; vl[7] += dz * fy_ij; vl[8] += dz * fz_ij;
            }
        }

#pragma omp atomic
        pot += pl;

#pragma omp critical
        {
            for (int j = 0; j < 9; ++j) vir[j] += vl[j];
        }
    }
}

} // namespace

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv)
{
    const int threads = parse_int_arg(argc, argv, "--threads", 1);
    const int natom = parse_int_arg(argc, argv, "--natoms", 2000000);
    const int repeat = parse_int_arg(argc, argv, "--repeat", 5);
    const int lj_atom = parse_int_arg(argc, argv, "--lj-natoms", 50000);
    const int lj_neigh = parse_int_arg(argc, argv, "--lj-neigh", 60);

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif

    Data d = make_data(natom, lj_neigh);
    Data dlj = make_data(lj_atom, lj_neigh);

    print_csv_header();

    // ---- 1. rescale_vel ----
    {
        auto s_vel = d.vel_backup;
        auto o_vel = d.vel_backup;
        rescale_vel_serial(d, s_vel);
        rescale_vel_omp(d, o_vel);
        double sms = time_repeated(repeat, [&] { rescale_vel_serial(d, s_vel); });
        double oms = time_repeated(repeat, [&] { rescale_vel_omp(d, o_vel); });
        print_measurement({"rescale_vel", threads, sms, oms,
                           max_diff_vec(s_vel, o_vel), checksum_vec(o_vel)});
    }

    // ---- 2. msst_vel_sum ----
    {
        double sr = msst_vel_sum_serial(d);
        double or_ = msst_vel_sum_omp(d);
        double sms = time_repeated(repeat, [&] { sink += msst_vel_sum_serial(d); });
        double oms = time_repeated(repeat, [&] { sink += msst_vel_sum_omp(d); });
        print_measurement({"msst_vel_sum", threads, sms, oms,
                           std::abs(sr - or_), or_});
    }

    // ---- 3. msst_propagate_vel ----
    {
        auto s_vel = d.vel_backup;
        auto o_vel = d.vel_backup;
        msst_propagate_vel_serial(d, s_vel);
        msst_propagate_vel_omp(d, o_vel);
        double sms = time_repeated(repeat, [&] { msst_propagate_vel_serial(d, s_vel); });
        double oms = time_repeated(repeat, [&] { msst_propagate_vel_omp(d, o_vel); });
        print_measurement({"msst_propagate_vel", threads, sms, oms,
                           max_diff_vec(s_vel, o_vel), checksum_vec(o_vel)});
    }

    // ---- 4. nhc_vel_baro ----
    {
        auto s_vel = d.vel_backup;
        auto o_vel = d.vel_backup;
        nhc_vel_baro_serial(d, s_vel);
        nhc_vel_baro_omp(d, o_vel);
        double sms = time_repeated(repeat, [&] { nhc_vel_baro_serial(d, s_vel); });
        double oms = time_repeated(repeat, [&] { nhc_vel_baro_omp(d, o_vel); });
        print_measurement({"nhc_vel_baro", threads, sms, oms,
                           max_diff_vec(s_vel, o_vel), checksum_vec(o_vel)});
    }

    // ---- 5. fire_check_fire (mix branch, P > 0) ----
    {
        auto s_vel = d.vel_backup;
        auto o_vel = d.vel_backup;
        auto sr = fire_check_fire_mix_serial(d, s_vel);
        auto or_ = fire_check_fire_mix_omp(d, o_vel);
        double sms = time_repeated(repeat, [&] { sink += fire_check_fire_mix_serial(d, s_vel).P; });
        double oms = time_repeated(repeat, [&] { sink += fire_check_fire_mix_omp(d, o_vel).P; });
        double diff = std::max({std::abs(sr.P - or_.P), std::abs(sr.sumforce - or_.sumforce),
                                std::abs(sr.normvel - or_.normvel), max_diff_vec(s_vel, o_vel)});
        print_measurement({"fire_check_fire_mix", threads, sms, oms, diff, checksum_vec(o_vel)});
    }

    // ---- 6. fire_check_fire (zero branch, P <= 0) ----
    {
        auto s_vel = d.vel_backup;
        auto o_vel = d.vel_backup;
        fire_check_fire_zero_serial(d, s_vel);
        fire_check_fire_zero_omp(d, o_vel);
        double sms = time_repeated(repeat, [&] { fire_check_fire_zero_serial(d, s_vel); });
        double oms = time_repeated(repeat, [&] { fire_check_fire_zero_omp(d, o_vel); });
        print_measurement({"fire_check_fire_zero", threads, sms, oms,
                           max_diff_vec(s_vel, o_vel), checksum_vec(o_vel)});
    }

    // ---- 7. lj_runner ----
    {
        auto sfx = dlj.lj_force_x, sofx = dlj.lj_force_x;
        auto sfy = dlj.lj_force_y, sofy = dlj.lj_force_y;
        auto sfz = dlj.lj_force_z, sozf = dlj.lj_force_z;
        double spot = 0, opot = 0;
        double svir[9] = {}, ovir[9] = {};
        lj_runner_serial(dlj, sfx, sfy, sfz, spot, svir);
        lj_runner_omp(dlj, sofx, sofy, sozf, opot, ovir);
        double sms = time_repeated(std::max(1, repeat / 2), [&] {
            lj_runner_serial(dlj, sfx, sfy, sfz, spot, svir);
        });
        double oms = time_repeated(std::max(1, repeat / 2), [&] {
            lj_runner_omp(dlj, sofx, sofy, sozf, opot, ovir);
        });
        double diff = std::abs(spot - opot);
        for (int j = 0; j < 9; ++j) diff = std::max(diff, std::abs(svir[j] - ovir[j]));
        diff = std::max({diff, max_diff_dbl(sfx, sofx), max_diff_dbl(sfy, sofy), max_diff_dbl(sfz, sozf)});
        print_measurement({"lj_runner", threads, sms, oms, diff, checksum_dbl(sofx)});
    }

    if (sink == -1.0) std::cerr << "unreachable\n";
    return 0;
}
