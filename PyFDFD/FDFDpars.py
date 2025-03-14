import PyFDFD.Forward_basic as Forward_basic
from base import PML,EquationType,FT,GT

def FDFDpars():
    SolverOpts = object()
    SolverOpts.pml = PML.sc
    SolverOpts.eqtype = EquationType(FT.e, GT.prim)
    # [osc, grid3d, s_factor, Eps, mu, J, M, M_S, ~, ~, ~, ~, ~] = build_system(solveropts.eqtype.ge, solveropts.pml, varargin{1:iarg}, pm, solveropts.returnAandb)
    # [A, b, ~, ~, ~] = create_eqTM(SolverOpts.eqtype, SolverOpts.pml, omega, Eps, mu, s_factor, J, M, grid3d)