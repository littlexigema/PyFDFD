from ..base import Axis
from .Source import Source
from .WithBloch import WithBloch
import torch
import math

class PlaneSrc(Source, WithBloch):
    """
    Plane Source class representing a constant electric dipole distribution over an entire plane.
    
    This class generates a plane wave in a homogeneous medium and supports oblique incidence.
    
    Args:
        normal_axis (Axis): Direction normal to the plane (Axis.X, Axis.Y, or Axis.Z)
        intercept (float): Location of the plane in the normal_axis direction
        polarization (Union[Axis, float]): Direction of the dipoles or angle in radians
        K (complex, optional): Amplitude of the surface current density. Defaults to 1.0
        theta (Union[float, list], optional): Oblique incidence angle or Bloch wavevector
        wvlen (float, optional): Wavelength in the background medium
    """
    
    def __init__(self, normal_axis, intercept, polarization, K=1.0, theta=None, wvlen=None):
        # Validate normal_axis
        if not isinstance(normal_axis, Axis):
            raise ValueError('"normal_axis" should be instance of Axis.')
            
        # Get cyclic permutation of axes
        p, q = self._cycle(normal_axis)
        
        # Validate intercept
        if not isinstance(intercept, (int, float)):
            raise ValueError('"intercept" should be real.')
            
        # Process polarization
        if not (isinstance(polarization, Axis) or isinstance(polarization, (int, float))):
            raise ValueError('"polarization" should be instance of Axis or angle in radian.')
            
        phi_ = polarization
        if isinstance(polarization, Axis):
            if polarization == normal_axis:
                raise ValueError('"polarization" should be orthogonal to "normal_axis".')
            phi_ = 0 if polarization == p else math.pi/2
            
        # Process K
        if not isinstance(K, complex):
            K = complex(K)
            
        # Process theta and wvlen
        if theta is None:
            kt_Bloch = torch.zeros(2)
        elif wvlen is None:
            kt_Bloch = theta  # theta is actually kt_Bloch
            if not (isinstance(kt_Bloch, torch.Tensor) and kt_Bloch.shape == (2,)):
                raise ValueError('"kt_Bloch" should be length-2 tensor with real elements.')
        else:
            if not (-math.pi/2 <= theta <= math.pi/2):
                raise ValueError('"theta" should be polar angle in radian between -pi/2 and pi/2.')
            if not (isinstance(wvlen, (int, float)) and wvlen > 0):
                raise ValueError('"wvlen" should be positive.')
                
            kt_Bloch = torch.zeros(2)
            kt = (2*math.pi/wvlen) * math.sin(theta)
            kp = kt * math.cos(phi_ + math.pi/2)
            kq = kt * math.sin(phi_ + math.pi/2)
            kt_Bloch[0] = kp
            kt_Bloch[1] = kq
            
        # Initialize parent classes
        lgrid = [None] * Axis.count()
        laltgrid = [None] * Axis.count()
        lgrid[normal_axis.value] = intercept
        super().__init__(lgrid, laltgrid)
        
        # Store properties
        self.normal_axis = normal_axis
        self.intercept = intercept
        self.phi = phi_
        self.K = K
        self.kBloch = torch.zeros(3)
        self.kBloch[[p.value, q.value]] = kt_Bloch
        
    def generate_kernel(self, w_axis, grid3d):
        """
        Generate the source kernel for the given axis and grid.
        
        Args:
            w_axis (Axis): Working axis
            grid3d (Grid3d): 3D grid object
            
        Returns:
            tuple: (index_cell, JMw_patch)
        """
        grid3d.set_kBloch(self)
        
        if w_axis == self.normal_axis:
            return [None] * Axis.count(), None
            
        n = self.normal_axis
        w = w_axis
        v = next(ax for ax in Axis.elems() if ax not in [n, w])
        
        g = self.gt
        ind_n = self._ind_for_loc(self.intercept, n, g, grid3d)
        dn = grid3d.dl[n.value, g][ind_n]
        J = self.K / dn
        
        if w == self._cycle(self.normal_axis)[0]:
            Jw = J * math.cos(self.phi)
        else:
            Jw = J * math.sin(self.phi)
            
        if Jw == 0:
            return [None] * Axis.count(), None
            
        lw = grid3d.l[w.value, self._alter(g)]
        lv = grid3d.l[v.value, g]
        kw = self.kBloch[w.value]
        kv = self.kBloch[v.value]
        
        # Generate JMw_patch using torch operations
        exp_w = torch.exp(-1j * (kw * lw)).unsqueeze(1)
        exp_v = torch.exp(-1j * (kv * lv)).unsqueeze(0)
        JMw_patch = Jw * (exp_w @ exp_v)
        
        # Permute dimensions to match MATLAB's ipermute
        perm = torch.tensor([w.value, v.value, n.value])
        JMw_patch = JMw_patch.permute(*torch.argsort(perm))
        
        # Set index_cell
        index_cell = [slice(None)] * 3
        index_cell[n.value] = ind_n
        
        return index_cell, JMw_patch