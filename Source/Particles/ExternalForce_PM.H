#ifndef WARPX_EXTERNALFORCE_PM_H
#define WARPX_EXTERNALFORCE_PM_H

#include <AMReX_REAL.H>
#include <PulsarParameters.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ApplyExternalForce(
    const amrex::Real xp, const amrex::Real yp, const amrex::Real zp,
    const amrex::Real ux, const amrex::Real uy, const amrex::Real uz,
    amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
    amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz,
    const amrex::Real xc, const amrex::Real yc, const amrex::Real zc,
    const amrex::Real q, const amrex::Real m, const amrex::Real wp, const long pid)
{
    /* physical properties of NS */

    // B_omega_sign = +/-1.0, B and Omega are parallel (+1) or antiparallel (-1)
    const amrex::Real B_omega_sign = PulsarParm::B_omega_alignment;

    // omega_star = rad/s, rotation axis is k_hat*B_omega_sign
    const amrex::Real omega_star = PulsarParm::omega_star;

    const amrex::Real r_star = PulsarParm::R_star; // meters
    const amrex::Real B_star = PulsarParm::B_star * B_omega_sign; // Tesla

    /* geometry */
    // spherical radius, theta, phi and cylindrical r //
    const amrex::Real r = std::sqrt( (xp-xc)*(xp-xc) + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc));
    const amrex::Real phi = std::atan2((yp-yc),(xp-xc));
    amrex::Real theta = 0.0;
    if (r>0) {
       theta = std::acos( (zp-zc)/r );
    }
    const amrex::Real r_cl = std::sin(theta) * r;

    amrex::Real c_theta, s_theta, c_phi, s_phi;

    c_theta = std::cos(theta);
    s_theta = std::sin(theta);
    c_phi = std::cos(phi);
    s_phi = std::sin(phi);

    constexpr amrex::Real inv_c2 = 1./(PhysConst::c*PhysConst::c);

    // Compute inverse Lorentz factor
    const amrex::Real inv_gamma = 1./std::sqrt(1. + (ux*ux + uy*uy + uz*uz)*inv_c2);

    /* print particle data before fields if pid is pid_verbose_external list */
    if (PulsarParm::verbose_external && pid == PulsarParm::pid_verbose_external) {
        amrex::Print() << "***** before ext force *** \n";
        amrex::Print() << " r " << r << "\n";
        amrex::Print() << " theta " << theta << "\n";
        amrex::Print() << " phi " << phi << "\n";
        amrex::Print() << " vel " << ux*inv_gamma << " " << uy*inv_gamma << " " << uz*inv_gamma << "\n";
        amrex::Print() << " xy speed " << std::sqrt(ux*ux*inv_gamma*inv_gamma + uy*uy*inv_gamma*inv_gamma ) << "\n";
        amrex::Print() << " r*omega " << r*s_theta*omega_star << "\n";;
    }

    /* Internal forces and fields */
    if (r <= r_star*1.001)
    {
        if (PulsarParm::use_drag_force) {
            // drag force quantities
            const amrex::Real f_drag_coeff = -1.0 * m /(q * PulsarParm::drag_force_tau);
            amrex::Real f_drag_ux, f_drag_uy, f_drag_uz;

            f_drag_ux = ux*inv_gamma - (-1.0)*(r_cl/r_star)*r_star*omega_star*std::sin(phi);
            f_drag_uy = uy*inv_gamma - (r_cl/r_star)*r_star*omega_star*std::cos(phi);
            f_drag_uz = uz*inv_gamma - 0.0;

            // Electric field equivalent to drag force
            Ex += f_drag_coeff * f_drag_ux ;
            Ey += f_drag_coeff * f_drag_uy ;
            Ez += f_drag_coeff * f_drag_uz;
        }

        if (PulsarParm::use_conductor_E) {
            // Electric field for a good conductor in steady state E = -(omega*rcyl)[cross]B
            Ex += (r_cl * omega_star) * (2.0 * B_star) * c_phi;
            Ey += (r_cl * omega_star) * (2.0 * B_star) * s_phi;
            Ez += 0.0;
        }

        // Uniform internal B field
        Bx += 0.0;
        By += 0.0;
        Bz += B_star * 2.0;
    }

    /* External forces and fields */
    if(r > r_star*1.001)
    {
        amrex::Real r_ratio = r_star/r;
        amrex::Real r3 = r_ratio*r_ratio*r_ratio;
        amrex::Real B_r = 2.0*B_star*r3*c_theta;
        amrex::Real B_theta = B_star*r3*s_theta;

        // Dipole B field
        Bx += B_r*c_phi*s_theta + B_theta*c_phi*c_theta;
        By += B_r*s_phi*s_theta + B_theta*s_phi*c_theta;
        Bz += B_r*c_theta - B_theta*s_theta;

        if (PulsarParm::use_external_E) {
            amrex::Real E_r = omega_star*r_star*B_star*r3*r_ratio*(1.0-3.0*c_theta*c_theta);

            if (PulsarParm::include_external_monopole_E)
                E_r += (2.0/3.0)*omega_star*B_star*r_star*r_ratio*r_ratio;

            amrex::Real E_theta = (-1.0)*omega_star*B_star*r_star*r3*r_ratio*(c_theta*s_theta*2.0);

            Ex += E_r*c_phi*s_theta + E_theta*c_phi*c_theta;
            Ey += E_r*s_phi*s_theta + E_theta*s_phi*c_theta;
            Ez += 1.0*E_r*c_theta - (1.0)*E_theta*s_theta;
        }
    }

    /* print particle data after fields if pid is pid_verbose_external list */
    if (PulsarParm::verbose_external && pid == PulsarParm::pid_verbose_external) {
        amrex::Print() << " **** after ext force **** \n" ;
        amrex::Print() << " Efield " << Ex << " " << Ey << " " << Ez << "\n";
        amrex::Print() << " Bfield " << Bx << " " << By << " " << Bz << "\n";
        amrex::Print() << " q " << q << "\n";
    }
}

#endif
