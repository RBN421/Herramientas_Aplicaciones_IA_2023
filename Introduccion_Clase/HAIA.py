import pandas as pd
import numpy as np
import string

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        self.dqr = self.get_dqr()
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
    
    def get_dqr(self):

        #% Lista de variables de la base de datos
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'], 
                               index=list(self.data.columns.values))

        #Lista de tipos de datos del dataframe
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])

        #Lista de valores presentes
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values'])

        #Lista de valores missing (Valores faltantes/nulos nan)
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values'])

        #Valores unicos de las columnas
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()]

        # Información estadística
        #Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max_values'])
        for col in list(self.data.columns.values):
            try:
                max_values.loc[col] = [self.data[col].max()]
            except:
                max_values.loc[col] = ['N/A']
                pass
        #Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            try:
                min_values.loc[col] = [self.data[col].min()]
            except:
                min_values.loc[col] = ['N/A']
                pass
        #Lista de valores con su desviación estandar
        #Lista de valores con los percentiles
        #Lista de valores con la media
        mean_values = pd.DataFrame(columns=['Mean_values'])
        for col in list(self.data.columns.values):
            try:
                mean_values.loc[col] = [self.data[col].mean()]
            except:
                mean_values.loc[col] = ['N/A']
                pass

        # Lista informativa que nos diga si una columna es categorica (True, False)
        is_categorical = pd.DataFrame(columns=['Is_Categorical'])
        cat_cols = self.data.select_dtypes(exclude='number').columns
        for col in list(self.data.columns.values):
            if col in cat_cols:
                is_categorical.loc[col] = True
            else:
                is_categorical.loc[col] = False

        # Si categorica es True ---> generar otra lista con las categorias de esas columnas


        return columns.join(data_dtypes).join(present_values).join(missing_values).join(unique_values).join(max_values).join(min_values).join(mean_values).join(is_categorical)
    
    ##% Métodos para limpieza de datos
    # Remover signos de puntuación
    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    # Remover digitos
    @staticmethod
    def remove_digits(x):
        try:
            x=''.join(ch for ch in x if ch not in string.digits)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    # remover espacios en blanco
    @staticmethod
    def remove_whitespace(x):
        try:
            x=' '.join(x.split())
        except:
            pass
        return x

    # convertir a minisculas
    @staticmethod
    def lower_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x

    #convertir a mayusculas
    @staticmethod
    def upper_text(x):
        try:
            x = x.upper()
        except:
            pass
        return x

    # Función que convierta a mayúsculas la primera letra,
    @staticmethod
    def capitalize_text(x):
        try:
            x = x.capitalize()
        except:
            pass
        return x
    # reemplazar texto
    @staticmethod
    def replace_text(x,to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x

  