#ifndef MD_STATISTICS_H
#define MD_STATISTICS_H

#include "source_base/matrix3.h"
#include "source_base/vector3.h"

/**
 * @brief 动能与温度统计结果 —— 纯数据结构
 *
 * 替代原来 current_temp(kinetic, natom, frozen_freedom, allmass, vel)
 * 中通过引用参数回写 kinetic 的方式。
 */
struct MDKineticState
{
    double kinetic     = 0.0; ///< 动能 (Hartree)
    double temperature = 0.0; ///< 温度 (Hartree)

    /// 方便转换为开尔文
    double temperature_kelvin(double hartree_to_k) const
    {
        return temperature * hartree_to_k;
    }
};

/**
 * @brief 应力统计结果 —— 纯数据结构
 *
 * 替代原来 compute_stress() 通过引用参数同时修改 virial 和 stress 的方式。
 * 将动能贡献张量 t_vector 和总应力 stress 分开返回。
 */
struct MDStressState
{
    ModuleBase::matrix t_vector; ///< 离子动能贡献张量 (3x3)
    ModuleBase::matrix stress;   ///< 总应力张量 = virial + t_vector/omega (3x3)
};

/**
 * @brief FIRE 优化器投影统计 —— 纯数据结构
 *
 * 替代原来 FIRE::check_fire() 中散落的 P、sumforce、normvel 局部变量。
 */
struct FIREProjection
{
    double power         = 0.0; ///< P = Σ v_i · f_i
    double force_norm    = 0.0; ///< |f| = sqrt(Σ |f_i|²)
    double velocity_norm = 0.0; ///< |v| = sqrt(Σ |v_i|²)
    double max_force     = 0.0; ///< max |f_i| component
};

#endif // MD_STATISTICS_H
