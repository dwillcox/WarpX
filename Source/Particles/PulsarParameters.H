#ifndef PULSAR_PARAMETERS_H
#define PULSAR_PARAMETERS_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace PulsarParm
{
    extern AMREX_GPU_DEVICE_MANAGED int use_conductor_E;
    extern AMREX_GPU_DEVICE_MANAGED int use_external_E;
    extern AMREX_GPU_DEVICE_MANAGED int include_external_monopole_E;
    extern AMREX_GPU_DEVICE_MANAGED int use_drag_force;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real drag_force_tau;

    extern AMREX_GPU_DEVICE_MANAGED int B_omega_alignment;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
    extern AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;

    extern AMREX_GPU_DEVICE_MANAGED int verbose_external;
    extern AMREX_GPU_DEVICE_MANAGED long pid_verbose_external;

    void Initialize();
}

#endif
