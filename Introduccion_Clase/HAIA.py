import pandas as pd
import numpy as np

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        
    ##% Métodos para Análisis de Datos 
    #Método para obtener las columnas y dataframe binarios
    def get_binarios(self):
        col_bin = []
        for col in self.columns:
            if self.data[col].nunique() == 2:
                col_bin.append(col)

        return self.data[col_bin], col_bin
    #Método para obtener columnas y dataframe cuantitativos
    def get_cuantitativos(self):
        col_cuantitativos = self.data.select_dtypes(include='number').columns
        
        return self.data[col_cuantitativos], col_cuantitativos
    #Método para obtener columnas y dataframe categóricos
    def get_categoricos(self):
        col_categorias = self.data.select_dtypes(exclude='number').columns
        col_cat=[]
        for col in col_categorias:
            if self.data[col].nunique()>2:
                col_cat.append(col)
        return self.data[col_cat], col_cat
  
