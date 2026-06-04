#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double& operator[](const int i)
    {
        if (i == 0)
        {
            return x;
        }
        if (i == 1)
        {
            return y;
        }
        return z;
    }

    const double& operator[](const int i) const
    {
        if (i == 0)
        {
            return x;
        }
        if (i == 1)
        {
            return y;
        }
        return z;
    }

    Vec3& operator*=(const double scale)
    {
        x *= scale;
        y *= scale;
        z *= scale;
        return *this;
    }
};

struct AtomType
{
    std::vector<Vec3> tau;
};

struct Fixture
{
    int nat = 0;
    int ntype = 0;
    double lat0_angstrom = 1.8897259886;
    std::vector<AtomType> atoms;
    std::vector<int> atom_type_index;
    std::vector<int> atom_local_index;
    std::vector<Vec3> velocity;
    std::vector<double> model_force;
};

struct KernelResult
{
    std::string kernel;
    int threads = 1;
    double serial_ms = 0.0;
    double omp_ms = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double max_abs_diff = 0.0;
};

struct Options
{
    int nat = 2000000;
    int repeats = 5;
    std::vector<int> threads = {1, 2, 4, 8};
};

double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int parse_int(const char* value, const char* name)
{
    const int parsed = std::atoi(value);
    if (parsed <= 0)
    {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return parsed;
}

std::vector<int> parse_threads(const std::string& text)
{
    std::vector<int> values;
    std::size_t start = 0;
    while (start < text.size())
    {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty())
        {
            values.push_back(parse_int(token.c_str(), "--threads"));
        }
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    if (values.empty())
    {
        throw std::runtime_error("--threads must contain at least one thread count");
    }
    return values;
}

Options parse_options(const int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--nat" && i + 1 < argc)
        {
            opt.nat = parse_int(argv[++i], "--nat");
        }
        else if (arg == "--repeats" && i + 1 < argc)
        {
            opt.repeats = parse_int(argv[++i], "--repeats");
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            opt.threads = parse_threads(argv[++i]);
        }
        else
        {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    return opt;
}

void set_threads(const int threads)
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

Fixture make_fixture(const int nat)
{
    Fixture fixture;
    fixture.nat = nat;
    fixture.ntype = 5;
    fixture.atoms.resize(fixture.ntype);
    fixture.atom_type_index.resize(nat);
    fixture.atom_local_index.resize(nat);
    fixture.velocity.resize(nat);
    fixture.model_force.resize(3 * static_cast<std::size_t>(nat));

    int remaining = nat;
    int iat = 0;
    for (int it = 0; it < fixture.ntype; ++it)
    {
        const int types_left = fixture.ntype - it;
        const int count = remaining / types_left;
        remaining -= count;
        fixture.atoms[it].tau.resize(count);
        for (int ia = 0; ia < count; ++ia)
        {
            const double base = static_cast<double>(iat + 1);
            fixture.atoms[it].tau[ia] = {0.001 * base + it, 0.002 * base - ia, 0.003 * base + ia * 0.5};
            fixture.atom_type_index[iat] = it;
            fixture.atom_local_index[iat] = ia;
            fixture.velocity[iat] = {1.0 + 0.00001 * base, -0.5 + 0.00002 * base, 0.25 - 0.00003 * base};
            fixture.model_force[3 * static_cast<std::size_t>(iat)] = 0.01 * base;
            fixture.model_force[3 * static_cast<std::size_t>(iat) + 1] = -0.02 * base;
            fixture.model_force[3 * static_cast<std::size_t>(iat) + 2] = 0.03 * base;
            ++iat;
        }
    }

    assert(iat == nat);
    return fixture;
}

void dpmd_coord_serial(const Fixture& fixture, std::vector<double>& coord)
{
    int iat = 0;
    for (int it = 0; it < fixture.ntype; ++it)
    {
        for (int ia = 0; ia < static_cast<int>(fixture.atoms[it].tau.size()); ++ia)
        {
            coord[3 * static_cast<std::size_t>(iat)] = fixture.atoms[it].tau[ia].x * fixture.lat0_angstrom;
            coord[3 * static_cast<std::size_t>(iat) + 1] = fixture.atoms[it].tau[ia].y * fixture.lat0_angstrom;
            coord[3 * static_cast<std::size_t>(iat) + 2] = fixture.atoms[it].tau[ia].z * fixture.lat0_angstrom;
            ++iat;
        }
    }
    assert(iat == fixture.nat);
}

void dpmd_coord_omp(const Fixture& fixture, std::vector<double>& coord)
{
    const int nat = fixture.nat;
#pragma omp parallel for schedule(static) if (nat >= 256)
    for (int iat = 0; iat < nat; ++iat)
    {
        const int it = fixture.atom_type_index[iat];
        const int ia = fixture.atom_local_index[iat];
        coord[3 * static_cast<std::size_t>(iat)] = fixture.atoms[it].tau[ia].x * fixture.lat0_angstrom;
        coord[3 * static_cast<std::size_t>(iat) + 1] = fixture.atoms[it].tau[ia].y * fixture.lat0_angstrom;
        coord[3 * static_cast<std::size_t>(iat) + 2] = fixture.atoms[it].tau[ia].z * fixture.lat0_angstrom;
    }
}

void dpmd_force_serial(const Fixture& fixture, std::vector<double>& force, const double fact_f)
{
    for (int i = 0; i < fixture.nat; ++i)
    {
        force[3 * static_cast<std::size_t>(i)] = fixture.model_force[3 * static_cast<std::size_t>(i)] * fact_f;
        force[3 * static_cast<std::size_t>(i) + 1] = fixture.model_force[3 * static_cast<std::size_t>(i) + 1] * fact_f;
        force[3 * static_cast<std::size_t>(i) + 2] = fixture.model_force[3 * static_cast<std::size_t>(i) + 2] * fact_f;
    }
}

void dpmd_force_omp(const Fixture& fixture, std::vector<double>& force, const double fact_f)
{
    const int nat_f = fixture.nat;
#pragma omp parallel for schedule(static) if (nat_f >= 256)
    for (int i = 0; i < nat_f; ++i)
    {
        force[3 * static_cast<std::size_t>(i)] = fixture.model_force[3 * static_cast<std::size_t>(i)] * fact_f;
        force[3 * static_cast<std::size_t>(i) + 1] = fixture.model_force[3 * static_cast<std::size_t>(i) + 1] * fact_f;
        force[3 * static_cast<std::size_t>(i) + 2] = fixture.model_force[3 * static_cast<std::size_t>(i) + 2] * fact_f;
    }
}

void scale_all_serial(std::vector<Vec3>& vel, const double scale)
{
    for (int i = 0; i < static_cast<int>(vel.size()); ++i)
    {
        vel[i] *= scale;
    }
}

void scale_all_omp(std::vector<Vec3>& vel, const double scale)
{
    const int nat = static_cast<int>(vel.size());
#pragma omp parallel for schedule(static) if (nat >= 256)
    for (int i = 0; i < nat; ++i)
    {
        vel[i] *= scale;
    }
}

void scale_component_serial(std::vector<Vec3>& vel, const int sd, const double scale)
{
    for (int i = 0; i < static_cast<int>(vel.size()); ++i)
    {
        vel[i][sd] *= scale;
    }
}

void scale_component_omp(std::vector<Vec3>& vel, const int sd, const double scale)
{
    const int nat = static_cast<int>(vel.size());
#pragma omp parallel for schedule(static) if (nat >= 256)
    for (int i = 0; i < nat; ++i)
    {
        vel[i][sd] *= scale;
    }
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    assert(lhs.size() == rhs.size());
    double diff = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
    }
    return diff;
}

double max_abs_diff(const std::vector<Vec3>& lhs, const std::vector<Vec3>& rhs)
{
    assert(lhs.size() == rhs.size());
    double diff = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        diff = std::max(diff, std::abs(lhs[i].x - rhs[i].x));
        diff = std::max(diff, std::abs(lhs[i].y - rhs[i].y));
        diff = std::max(diff, std::abs(lhs[i].z - rhs[i].z));
    }
    return diff;
}

template <class Func>
double average_ms(const int repeats, Func&& func)
{
    double total = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        const double start = now_ms();
        func();
        total += now_ms() - start;
    }
    return total / repeats;
}

template <class ForwardFunc, class ReverseFunc>
double average_pair_ms(const int repeats, ForwardFunc&& forward_func, ReverseFunc&& reverse_func)
{
    double total = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        const double start = now_ms();
        forward_func();
        reverse_func();
        total += now_ms() - start;
    }
    return total / (2.0 * repeats);
}

KernelResult make_result(const std::string& kernel,
                         const int threads,
                         const double serial_ms,
                         const double omp_ms,
                         const double max_diff)
{
    KernelResult result;
    result.kernel = kernel;
    result.threads = threads;
    result.serial_ms = serial_ms;
    result.omp_ms = omp_ms;
    result.speedup = serial_ms / omp_ms;
    result.efficiency = result.speedup / threads;
    result.max_abs_diff = max_diff;
    return result;
}

KernelResult benchmark_dpmd_coord(const Fixture& fixture, const int threads, const int repeats)
{
    std::vector<double> serial_coord(3 * static_cast<std::size_t>(fixture.nat));
    std::vector<double> omp_coord(3 * static_cast<std::size_t>(fixture.nat));

    dpmd_coord_serial(fixture, serial_coord);
    set_threads(threads);
    dpmd_coord_omp(fixture, omp_coord);
    const double diff = max_abs_diff(serial_coord, omp_coord);

    const double serial_ms = average_ms(repeats, [&]() { dpmd_coord_serial(fixture, serial_coord); });
    set_threads(threads);
    const double omp_ms = average_ms(repeats, [&]() { dpmd_coord_omp(fixture, omp_coord); });
    return make_result("dpmd_coord_fill", threads, serial_ms, omp_ms, diff);
}

KernelResult benchmark_dpmd_force(const Fixture& fixture, const int threads, const int repeats)
{
    constexpr double fact_f = 0.12345;
    std::vector<double> serial_force(3 * static_cast<std::size_t>(fixture.nat));
    std::vector<double> omp_force(3 * static_cast<std::size_t>(fixture.nat));

    dpmd_force_serial(fixture, serial_force, fact_f);
    set_threads(threads);
    dpmd_force_omp(fixture, omp_force, fact_f);
    const double diff = max_abs_diff(serial_force, omp_force);

    const double serial_ms = average_ms(repeats, [&]() { dpmd_force_serial(fixture, serial_force, fact_f); });
    set_threads(threads);
    const double omp_ms = average_ms(repeats, [&]() { dpmd_force_omp(fixture, omp_force, fact_f); });
    return make_result("dpmd_force_copy_back", threads, serial_ms, omp_ms, diff);
}

KernelResult benchmark_scale_all(const std::string& kernel,
                                 const Fixture& fixture,
                                 const int threads,
                                 const int repeats,
                                 const double scale)
{
    std::vector<Vec3> serial_vel = fixture.velocity;
    std::vector<Vec3> omp_vel = fixture.velocity;

    scale_all_serial(serial_vel, scale);
    set_threads(threads);
    scale_all_omp(omp_vel, scale);
    const double diff = max_abs_diff(serial_vel, omp_vel);

    const double inv_scale = 1.0 / scale;
    const double serial_ms = average_pair_ms(repeats,
                                             [&]() { scale_all_serial(serial_vel, scale); },
                                             [&]() { scale_all_serial(serial_vel, inv_scale); });
    set_threads(threads);
    const double omp_ms = average_pair_ms(repeats,
                                          [&]() { scale_all_omp(omp_vel, scale); },
                                          [&]() { scale_all_omp(omp_vel, inv_scale); });
    return make_result(kernel, threads, serial_ms, omp_ms, diff);
}

KernelResult benchmark_scale_component(const Fixture& fixture, const int threads, const int repeats)
{
    constexpr int sd = 1;
    constexpr double dilation = 0.9975;
    std::vector<Vec3> serial_vel = fixture.velocity;
    std::vector<Vec3> omp_vel = fixture.velocity;

    scale_component_serial(serial_vel, sd, dilation);
    set_threads(threads);
    scale_component_omp(omp_vel, sd, dilation);
    const double diff = max_abs_diff(serial_vel, omp_vel);

    const double inv_dilation = 1.0 / dilation;
    const double serial_ms = average_pair_ms(repeats,
                                             [&]() { scale_component_serial(serial_vel, sd, dilation); },
                                             [&]() { scale_component_serial(serial_vel, sd, inv_dilation); });
    set_threads(threads);
    const double omp_ms = average_pair_ms(repeats,
                                          [&]() { scale_component_omp(omp_vel, sd, dilation); },
                                          [&]() { scale_component_omp(omp_vel, sd, inv_dilation); });
    return make_result("msst_rescale_velocity_component", threads, serial_ms, omp_ms, diff);
}

void print_header()
{
    std::cout << "kernel,threads,serial_ms,omp_ms,speedup,efficiency,max_abs_diff\n";
}

void print_result(const KernelResult& result)
{
    std::cout << result.kernel << ',' << result.threads << ',' << std::fixed << std::setprecision(6)
              << result.serial_ms << ',' << result.omp_ms << ',' << result.speedup << ','
              << result.efficiency << ',' << std::scientific << result.max_abs_diff << std::defaultfloat << '\n';
}

} // namespace

int main(const int argc, char** argv)
{
    try
    {
        const Options opt = parse_options(argc, argv);
        const Fixture fixture = make_fixture(opt.nat);

        print_header();
        for (const int threads : opt.threads)
        {
            print_result(benchmark_dpmd_coord(fixture, threads, opt.repeats));
            print_result(benchmark_dpmd_force(fixture, threads, opt.repeats));
            print_result(benchmark_scale_all("verlet_thermalize_velocity", fixture, threads, opt.repeats, 1.0125));
            print_result(benchmark_scale_component(fixture, threads, opt.repeats));
            print_result(benchmark_scale_all("nhc_particle_thermo_velocity", fixture, threads, opt.repeats, 0.99875));
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
