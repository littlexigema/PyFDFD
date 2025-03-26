import torch

class Guess:
    """
    微波电磁反演，对epsil的猜测，包括\chi,J等
    """
    def __init__(self):
        self._epsil = None
        self._J = None

    def set_epsil(self,epsil:torch.Tensor):
        self._epsil = epsil

    def set_J(self,J:torch.Tensor):
        self._J = J
    @property
    def epsil(self):
        return self._epsil
    @property
    def J(self):
        return self._J
    