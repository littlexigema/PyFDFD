from PyFDFD.shape.Box import Box
from PyFDFD.build_system import build_system
from PyFDFD.material.Material import Material
from PyFDFD.io.EMObject import EMObject
from PyFDFD.base.GT import GT
from typing import Union
from config import *
from Field import Field
import numpy as np

class Forward_model:
    m_unit = m_unit
    xlchi           = round(invdom[0] / m_unit)
    xhchi           = round(invdom[1] / m_unit)
    ylchi           = round(invdom[2] / m_unit)
    yhchi           = round(invdom[3] / m_unit)
    zlchi           = round(invdom[4] / m_unit)
    zhchi           = round(invdom[5] / m_unit)
    def __init__(self):
        self.dl = 1
        tmp = np.array([[Forward_model.xlchi,Forward_model.xhchi],[Forward_model.ylchi,Forward_model.yhchi],[0,1]])
        self.domain = Box(tmp,self.dl)
        self.material = Material('vacuum', 'none', 1.0)
        self.emobj = EMObject([self.domain],self.material)
        self.field = Field()
    def get_system_matrix(self,fre:Union[int,float]):
        _,wvlen = Field.get_lambda(fre)#离散波长
        omega = 2*pi/wvlen
        A_for,A_back = build_system(Forward_model.m_unit,wvlen,self.domain,Field.Lpml,self.emobj)
        E_inc = self.field.get_incident_field()


if __name__=="__main__":
    FWD = Forward_model()
    FWD.get_system_matrix(fre)