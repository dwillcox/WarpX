#include <PulsarParameters.H>
#include <AMReX_ParmParse.H>

namespace PulsarParm
{
    AMREX_GPU_DEVICE_MANAGED int use_conductor_E = 0; // [0/1]: set E = -(omega*R)[cross]B inside NS
    AMREX_GPU_DEVICE_MANAGED int use_external_E = 0;  // [0/1]: use external analytic E field
    AMREX_GPU_DEVICE_MANAGED int include_external_monopole_E = 0; // [0/1]: turn off/on external E monopole term
    AMREX_GPU_DEVICE_MANAGED int use_drag_force = 1;  // [0/1]: apply drag force within the NS
    AMREX_GPU_DEVICE_MANAGED amrex::Real drag_force_tau = 1.0e-8; // drag force timescale (seconds)

    AMREX_GPU_DEVICE_MANAGED int B_omega_alignment = 1; // [1/-1]: sign of B[dot]omega

    AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star = 1.e4; // angular velocity of NS (rad/s)
    AMREX_GPU_DEVICE_MANAGED amrex::Real R_star = 10.e3; // radius of NS (m)
    AMREX_GPU_DEVICE_MANAGED amrex::Real B_star = 1.e8; // magnetic field of NS (T)

    AMREX_GPU_DEVICE_MANAGED int verbose_external = 0; // [0/1]: turn on verbosity for external force
    AMREX_GPU_DEVICE_MANAGED long pid_verbose_external = 0; // print particle properties for this particle ID if verbose_external = 1.

    void Initialize()
    {
        amrex::ParmParse pp("pulsar");

        pp.query("use_conductor_E", use_conductor_E);
        pp.query("use_external_E", use_external_E);
        pp.query("include_external_monopole_E", include_external_monopole_E);
        pp.query("use_drag_force", use_drag_force);
        pp.query("drag_force_tau", drag_force_tau);

        pp.query("B_omega_alignment", B_omega_alignment);
        pp.query("omega_star", omega_star);
        pp.query("R_star", R_star);
        pp.query("B_star", B_star);

        pp.query("verbose_external", verbose_external);
        pp.query("pid_verbose_external", pid_verbose_external);
    }
}
