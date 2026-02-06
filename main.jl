"""
    main.jl — SGSD-DEED Experiment Entry Point

    Usage:
      using Pkg; Pkg.activate(".")
      include("main.jl")

    To reproduce ALL paper results:
      run_paper_experiments()

    Or run individual experiments:
      run_drdeed_comparison()        # DR-DEED comparison
      run_scalability()              # Scalability analysis
      run_ieee30_validation()        # IEEE 30-bus DC-OPF
      run_model_progression()        # Model progression
      run_sensitivity()              # Sensitivity analysis
      run_saudi_case_study()         # Saudi case study
      run_pareto_analysis()          # Pareto front (ε-constraint)
      run_metaheuristic_comparison() # NSGA-II vs Ipopt
"""

include("experiments.jl")
