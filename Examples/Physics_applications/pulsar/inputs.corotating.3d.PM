#################################
####### GENERAL PARAMETERS ######
#################################
max_step = 99000
amr.n_cell = 128 128 128
amr.max_grid_size = 64
amr.blocking_factor = 64
amr.max_level = 0
amr.plot_file = "plotfiles/plt"
amr.plot_int = 100
geometry.coord_sys   = 0                  # 0: Cartesian
geometry.is_periodic = 0 0 0     # Is periodic?
#geometry.prob_lo     = -48000. -48000. -48000.
#geometry.prob_hi     =  48000.  48000.  48000.
geometry.prob_lo     =  0. 0. 0.
geometry.prob_hi     =  96.e3  96.e3  96.e3

#################################
############ NUMERICS ###########
#################################
algo.maxwell_fdtd_solver = yee
warpx.verbose = 1
warpx.plot_raw_fields = 0
warpx.do_dive_cleaning = 0
warpx.use_filter = 1
warpx.cfl = .99
my_constants.pi    = 3.141592653589793
my_constants.dens  = 1.0e20
my_constants.xc  = 48.e3
my_constants.yc  = 48.e3 
my_constants.zc  = 48.e3
my_constants.r_star = 12.e3
my_constants.omega = 6283.0
my_constants.c = 299792458. 
interpolation.nox = 3
interpolation.noy = 3
interpolation.noz = 3

#################################
############ PLASMA #############
#################################
particles.nspecies = 2
particles.species_names = plasma_e plasma_p

plasma_e.charge = -q_e
plasma_e.mass = m_e
plasma_e.injection_style = "NRandomPerCell"
# plasma_e.profile = constant
# plasma_e.density = 1.e20
plasma_e.profile = parse_density_function
plasma_e.density_function(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*dens"
plasma_e.num_particles_per_cell = 50
plasma_e.momentum_distribution_type = parse_momentum_function
plasma_e.momentum_function_ux(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(-omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (y-yc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_e.momentum_function_uy(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (x-xc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_e.momentum_function_uz(x,y,z) = "0.0"
plasma_e.do_continuous_injection = 0
plasma_e.density_min = 1E0

plasma_p.charge = q_e
plasma_p.mass = m_e
plasma_p.injection_style = "NRandomPerCell"
# plasma_p.profile = "constant"
# plasma_p.density = 1.e20
plasma_p.profile = parse_density_function
plasma_p.density_function(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*dens"
plasma_p.num_particles_per_cell = 50
plasma_p.momentum_distribution_type = parse_momentum_function
plasma_p.momentum_function_ux(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(-omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (y-yc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_p.momentum_function_uy(x,y,z) = "( (( (z-zc)*(z-zc) + (y-yc)*(y-yc) + (x-xc)*(x-xc) )^(0.5))<r_star)*(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/((1.0-(omega*(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5))/c)^2)^(0.5)) * (x-xc)/(((x-xc)*(x-xc) + (y-yc)*(y-yc))^(0.5)))"
plasma_p.momentum_function_uz(x,y,z) = "0.0"
plasma_p.do_continuous_injection = 0
plasma_p.density_min = 1E0

#################################
######### PULSAR SETUP ##########
#################################
pulsar.use_conductor_E = 1             # [0/1]: set E = -(omega*R)[cross]B inside NS
pulsar.use_external_E = 0              # [0/1]: use external analytic E field
pulsar.include_external_monopole_E = 0 # [0/1]: turn off/on external E monopole term
pulsar.use_drag_force = 0              # [0/1]: apply drag force within the NS
pulsar.drag_force_tau = 1.0e-16        # drag force timescale (seconds)

pulsar.B_omega_alignment = 1           # [1/-1]: sign of B[dot]omega
pulsar.omega_star = 6283               # angular velocity of NS (rad/s)
pulsar.R_star = 12.e3                  # radius of NS (m)
pulsar.B_star = 1.0e8                  # magnetic field of NS (T)

pulsar.verbose = 0                     # [0/1]: turn on verbosity for debugging print statements
pulsar.pid_verbose = 11264000          # print particle properties for this particle ID if pulsar.verbose = 1

