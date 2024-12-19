from PyFDFD.base import Box
from PyFDFD import build_system
from config import *
from Field import Field
import numpy as np

class Forward_model:
    m_unit = m_unit
    xlchi           = round(invdom(1) / m_unit)
    xhchi           = round(invdom(2) / m_unit)
    ylchi           = round(invdom(3) / m_unit)
    yhchi           = round(invdom(4) / m_unit)
    zlchi           = round(invdom(5) / m_unit)
    zhchi           = round(invdom(6) / m_unit)
    def __init__(self):
        self.domain = Box(np.array([Forward_model.xlchi,Forward_model.xhchi],[Forward_model.ylchi,Forward_model.yhchi],[Forward_model.zlchi,Forward_model.zhchi]))
        self.dl = 1
        self.field = Field()
    def get_system_matrix(self):
        M_s, A, b = build_system(self.domain,Field.Lpml)