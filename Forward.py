from PyFDFD.shape.Box import Box
from PyFDFD.Forward_basic import Forward_basic
from PyFDFD.material.Material import Material
from PyFDFD.io.EMObject import EMObject
from PyFDFD.base import Axis
from PyFDFD.base.GT import GT
from Guess import Guess
from typing import Union
from config import *
from Field import Field
import numpy as np
import torch

class Forward_model(Forward_basic):
    # m_unit = m_unit
    xlchi           = round(invdom[0] / m_unit)
    xhchi           = round(invdom[1] / m_unit)
    ylchi           = round(invdom[2] / m_unit)
    yhchi           = round(invdom[3] / m_unit)
    zlchi           = round(invdom[4] / m_unit)
    zhchi           = round(invdom[5] / m_unit)
    def __init__(self,inverse:bool = False):
        """
        inverse = True代表是反演过程的前向模型
                = False代表合成数据的前向模型
        """
        self.dl = 1
        self.inverse = inverse
        if inverse:
            tmp = np.array([[Forward_model.xlchi,Forward_model.xhchi],[Forward_model.ylchi,Forward_model.yhchi],[0,1]])
            self.domain = Box(tmp,self.dl)
            self.m_unit = m_unit
        else:
            tmp = [round(ele/m_unit_for) for ele in fordom]+[0,1]
            tmp = np.array(tmp).reshape(Axis.count(),-1)
            self.domain = Box(tmp,self.dl)
            self.m_unit = m_unit_for
        self.material = Material('vacuum', 'none', 1.0)
        self.emobj = EMObject([self.domain],self.material)
        self.field = Field()
        # self.field.set_incident_E_MOM()
        # if not inverse:
        #     """FDFD合成数据"""
        #     self.create_eqTM()#获得系统矩阵A和激励b
        #     self.field.set_chi(load_from_gt=True,**{'file':file_path})
        # self.field.set_scatter_E(omega)
        self.guess = Guess()
    def get_system_matrix(self,fre:Union[int,float],srcj_array:list=list()):
        """构建\epsilon背景系统矩阵和激励源矩阵"""
        _,wvlen = Field.get_lambda(fre)#离散波长
        # omega = 2*pi/wvlen
        A_for,A_back,b = self.build_system(self.m_unit,wvlen,self.domain,Field.Lpml,self.emobj,srcj_array=srcj_array)
        self.A_for = A_for
        self.ind = int(self.A_for.size(0)/3)
        A = (A_for+A_back).to_dense()
        self.A = A[2*self.ind:,2*self.ind:]
        self.b = b[2*self.ind:,]
        # return A[2*self.ind:,2*self.ind:]
    def get_system_matrix_epsil(self):
        """从FDFD原理上来说，不需要调用这个函数"""
        # _,wvlen = Field.get_lambda(fre)#离散波长
        A_back = self.build_system_back()
        A = (self.A_for+A_back).to_dense()
        self.A = A[2*self.ind:,2*self.ind:]
        # return A[2*self.ind:,2*self.ind:]
        # E_inc = self.field.get_incident_field()


if __name__=="__main__":
    FWD = Forward_model()
    FWD.get_system_matrix(fre)
    FWD.get_system_matrix_epsil(fre)