from abc import ABC, abstractproperty

class WithBloch(ABC):
    """
    Abstract class inherited by source classes that can set the Bloch boundary condition.
    Similar to MATLAB's multiple inheritance pattern.
    """
    
    def __init__(self):
        """
        Initialize WithBloch base class.
        """
        self._kBloch = None

    @property
    def kBloch(self):
        """
        Get Bloch wave vector.
        """
        return self._kBloch

    @kBloch.setter
    def kBloch(self, value):
        """
        Set Bloch wave vector.
        
        Args:
            value (torch.Tensor): The Bloch wave vector to set
        """
        self._kBloch = value