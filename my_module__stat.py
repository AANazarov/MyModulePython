# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 00:42:31 2022

@author: Alexander A. Nazarov
"""


###############################################################################
#           СТАТИСТИЧЕСКИЙ АНАЛИЗ / STATISTICAL ANALYSIS
#------------------------------------------------------------------------------
#           ПОЛЬЗОВАТЕЛЬСКИЕ СТАТИСТИЧЕСКИЕ ФУНКЦИИ
#           CUSTOM STATISTICAL FUNCTIONS
###############################################################################

import time
start_time = time.time()

#%clear    # команда очистки консоли


CALCULATION_VERSION = 220709    # ВЕРСИЯ РАСЧЕТА


#==============================================================================
#               ПОДКЛЮЧЕНИЕ МОДУЛЕЙ И БИБЛИОТЕК
#==============================================================================

import os
import sys
import platform

import math    # модуль доступа к математическим функциям
from math import *    # подключаем все содержимое модуля math, используем без псевдонимов

import numpy as np
from numpy import nan

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import scipy as sci
import scipy.stats as sps

import statsmodels.api as sm

import statistics as stat    # module 'statistics' has no attribute '__version__'

import sympy as sym


#=============================================================================
#               КОНСТАНТЫ
#=============================================================================

INCH = 25.4                                                                     # мм/дюйм
NUMBER_CHAR_LINE = 79    # число символов в строке
TABLE_HEAD_1 = ('#' * NUMBER_CHAR_LINE)
TABLE_HEAD_2 = ('=' * NUMBER_CHAR_LINE)
TABLE_HEAD_3 = ('-' * NUMBER_CHAR_LINE)


#=============================================================================
#               НАСТРОЙКА ШРИФТОВ В ГРАФИКАХ
#=============================================================================

f_size = 8    # пользовательская переменная для задания базового размера шрифта


#==============================================================================
#               ФУНКЦИИ ДЛЯ ПЕРВИЧНОЙ ОБРАБОТКИ ДАННЫХ
#==============================================================================

'''print(TABLE_HEAD_2)
Title = "1. ФУНКЦИИ ДЛЯ ПЕРВИЧНОЙ ОБРАБОТКИ ДАННЫХ"
print(' ' * int((NUMBER_CHAR_LINE - len(Title))/2), Title)
print(TABLE_HEAD_2, 2*'\n')'''


#------------------------------------------------------------------------------
#   Функция df_system_memory
#
#   Возвращает объем памяти, занимаемой DataFrame (на системном уровне)
#------------------------------------------------------------------------------

def df_system_memory(df_in, measure_unit='MB', digit=2, detailed=True):
    df_in_memory_usage = df_in.memory_usage(deep=True)
    if detailed:
        display(df_in_memory_usage)        
    
    df_in_memory_usage_sum = df_in_memory_usage.sum()
    if measure_unit=='MB':
        return round(df_in_memory_usage_sum / (2**20), digit)
    elif measure_unit=='B':
        return round(df_in_memory_usage_sum, 0)
    
# Тест функции

'''data = pd.DataFrame(np.random.rand(3, 2),
                    columns=['foo', 'bar'],
                    index=['a', 'b', 'c'])

print(f"data = \n{data}\n {type(data)}\n")
memory_1 = df_system_memory(data, digit=3)
print(f"Объем занимаемой памяти (на системном уровне): {memory_1} MB")'''


#------------------------------------------------------------------------------
#   Функция unique_values
#
#   Принимает в качестве аргумента **DataFrame** и возвращает 
#   другой **DataFrame** со следующей структурой:
#    * индекс - наименования столбцов из исходного **DataFrame**
#    * столбец **Num_Unique** - число уникальных значений
#    * столбец **Type** - тип данных
#    * столбец **Memory_usage (MB)** - объем занимаемой памяти на системном 
#      уровне (в МБ)
#------------------------------------------------------------------------------

def unique_values(df_in, sorting=True):
    df_out = pd.DataFrame(
        np.transpose(
            [df_in.nunique(),
             df_in.dtypes,
             round(df_in.memory_usage(
                 deep=True).iloc[1:len(df_in)] / (2**20), 2)]),
        index=df_in.columns,
        columns=['Num_Unique', 'Type', 'Memory_usage (MB)'])
    
    if sorting:
        return df_out.sort_values(by='Num_Unique', ignore_index=False)
    else:
        return df_out


#------------------------------------------------------------------------------
#   Функция transformation_to_category
#
#   Принимает в качестве аргумента **DataFrame** и преобразует выбранные 
#    признаки в категориальные
#------------------------------------------------------------------------------  

def transformation_to_category(df_in, max_unique_count=150, cols_to_exclude=None):
    # количество уникальных значений по столбцам исходной таблицы
    df_number_unique_values = unique_values(df_in, sorting=False)
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! АРГУМЕНТ ПО УМОЛЧАНИЮ
    cols_to_exclude = cols_to_exclude or []
    
    for col in df_in.columns:
        # проверяем условие
        if (df_in[col].nunique() < max_unique_count) and (col not in cols_to_exclude):
            df_in[col] = df_in[col].astype('category')    # преобразуем тип столбца

    return df_in  



#------------------------------------------------------------------------------
#   Функция df_detection_values
#------------------------------------------------------------------------------  

def df_detection_values(
    df_in,
    detection_values=[nan],
    graph_size=(210/INCH, 297/INCH/2)):
    
    cols = df_in.columns
    detection_values_df = df_in.isin(detection_values)
    
    col_size = np.array([df_in[elem].size for elem in cols])
    col_detection_values_count = [detection_values_df[elem].sum() for elem in cols]
    result_df = pd.DataFrame(
        {'size': col_size,
         'detected values': col_detection_values_count,
         'percentage of detected values': col_detection_values_count/col_size},
        index=df_in.columns)
    
    fig, axes = plt.subplots(figsize=graph_size)
    sns.heatmap(
        detection_values_df,
        cmap=sns.color_palette(['grey', 'yellow']),
        cbar=False,
        ax=axes)
    plt.show()
       
    return result_df, detection_values_df


#==============================================================================
#               ФУНКЦИИ ДЛЯ ОПИСАТЕЛЬНОЙ СТАТИСТИКИ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция descriptive_characteristics
#
#   Принимает в качестве аргумента np.ndarray и возвращает DataFrame,
#   содержащий основные статистические характеристики с доверительными 
#   интервалами и ошибками определения - расширенный аналог describe
#------------------------------------------------------------------------------

def descriptive_characteristics(
    X,
    p_level=0.95,
    auxiliary_table: bool = False):
    
    a_level = 1 - p_level
    
    # Вспомогательные величины
    #-------------------------
    
    # распределение Стьюдента
    f_t = len(X)-1    # число степеней свободы для t-квантиля
    gamma_t = 1 - a_level/2    # довер.вер-ть для t-квантиля
    t_p = sps.t.ppf(gamma_t, f_t)    # табл.значение t-квантиля
    
    # нормальное распределение
    u_p = sps.norm.ppf(p_level, 0, 1)    # табл.значение квантиля норм.распр.
    
    # распределение хи-квадрат
    f_chi2 = len(X) - 1    # число степеней свободы хи2
    gamma_chi2_low = (1 + p_level)/2    # довер.вер-ть хи2 нижн.
    gamma_chi2_up = (1 - p_level)/2    # довер.вер-ть хи2 верх.
    chi2_p_low = sps.chi2.ppf(gamma_chi2_low, f_chi2)
    chi2_p_up = sps.chi2.ppf(gamma_chi2_up, f_chi2)
    
    # Расчет статистических характеристик
    #------------------------------------
    
    # объем выборки
    N = round(len(X), 0)
    
    # среднее арифметическое
    X_mean = X.mean()
    X_std = X.std(ddof = 1)
    conf_int_low_X_mean = X_mean - sps.t.ppf(gamma_t, f_t) * X_std/math.sqrt(N)
    conf_int_up_X_mean = X_mean + sps.t.ppf(gamma_t, f_t) * X_std/math.sqrt(N)
    abs_err_X_mean = X_std / math.sqrt(N)
    rel_err_X_mean = abs_err_X_mean / X_mean * 100
    
    # медиана
    Me = np.median(X)
    C_p_Me = floor((N +  1 - u_p*sqrt(N + 0.5 - 0.25*u_p**2))/2)   # вспом.величина
    X_sort = np.array(sorted(X))
    conf_int_low_Me = X_sort[C_p_Me-1]
    conf_int_up_Me = X_sort[(N - C_p_Me + 1) - 1]
    abs_err_Me = abs_err_X_mean * sqrt(pi/2)
    rel_err_Me = abs_err_Me / Me * 100
    # довер.интервал для медианы - см. ГОСТ Р 50779.24-2005, п.6.2-6.3
    if X_mean < Me:
        Me_note = 'distribution is negative skewed (левосторонняя асимметрия) (mean < median)'
    else:
        Me_note = 'distribution is positive skewed (правосторонняя асимметрия) (mean > median)'
    
    # мода        
    Mo = stat.mode(X)
    
    # выборочная дисперсия
    D_X = np.var(X, ddof = 1)
    conf_int_low_D_X = D_X * (N - 1)/chi2_p_low
    conf_int_up_D_X = D_X * (N - 1)/chi2_p_up
    abs_err_D_X = D_X / sqrt(2*N)
    rel_err_D_X = abs_err_D_X / D_X * 100
    
    # выборочное С.К.О.
    conf_int_low_X_std = sqrt(conf_int_low_D_X)
    conf_int_up_X_std = sqrt(conf_int_up_D_X)
    abs_err_X_std = X_std / sqrt(2*N)
    rel_err_X_std = abs_err_X_std / X_std * 100   
    
    # выборочное С.Л.О.
    mad_X = pd.Series(X).mad()
    
    # минимальное, максимальное значения и вариационный размах
    min_X = np.amin(X)
    max_X = np.amax(X)
    R = np.ptp(X)
    
    # выборочные квантили
    np.percentile(X, 25)
    np.percentile(X, 50)
    np.percentile(X, 75)
    IQR = np.percentile(X, 75) - np.percentile(X, 25)
    np.percentile(X, 5)
    np.percentile(X, 95)
    
    # коэффициент вариации
    CV = sps.variation(X)
    conf_int_low_CV = CV / (1 + u_p/sqrt(2*(N - 1)) * sqrt(1 + 2*CV**2))
    conf_int_up_CV = CV / (1 - u_p/sqrt(2*(N - 1)) * sqrt(1 + 2*CV**2))
    abs_err_CV = CV / sqrt(N - 1) * sqrt(0.5 + CV**2)
    rel_err_CV = abs_err_CV / CV * 100
    if CV <= 0.33:
        CV_note = 'CV <= 0.33 (homogeneous population)'
    else:
        CV_note = 'CV > 0.33 (heterogeneous population)'
    
    # квартильный коэффициент дисперсии
    QCD = (np.percentile(X, 75) - np.percentile(X, 25)) / (np.percentile(X, 75) + np.percentile(X, 25))
    
    # показатель асимметрии
    As = sps.skew(X)
    abs_err_As = sqrt(6*N*(N-1) / ((N-2)*(N+1)*(N+3)))
    rel_err_As = abs_err_As / As * 100
    
    if abs(As) <= 0.25:
        As_note = 'distribution is approximately symmetric (распределение приблизительно симметричное) (abs(As)<=0.25)'
    elif abs(As) <= 0.5:
        if As < 0:
            As_note = 'distribution is moderately negative skewed (умеренная левосторонняя асимметрия) (abs(As)<=0.5, As<0)'
        else:
            As_note = 'distribution is moderately positive skewed (умеренная правосторонняя асимметрия) (abs(As)<=0.5, As>0)'
    else:
        if As < 0:
            As_note = 'distribution is highly negative skewed (значительная левосторонняя асимметрия) (abs(As)>0.5, As<0)'
        else:
            As_note = 'distribution is highly positive skewed (значительная правосторонняя асимметрия) (abs(As)>0.5, As>0)'
            
    # показатель эксцесса
    Es = sps.kurtosis(X)
    abs_err_Es = sqrt(24*N*(N-1)**2 / ((N-3)*(N-2)*(N+3)*(N+5)))
    rel_err_Es = abs_err_Es / Es * 100
    if Es > 0:
        Es_note = 'leptokurtic distribution (островершинное распределение) (Es>0)'
    elif Es < 0:
        Es_note = 'platykurtic distribution (плосковершинное распределение) (Es<0)'
    else:
        Es_note = 'mesokurtic distribution (нормальное распределение) (Es=0)'
    
    # Создадим DataFrame для сводки результатов
    #------------------------------------------
    
    # основная таблица
    result = pd.DataFrame({
        'characteristic': (
            'count', 'mean', 'median', 'mode',
            'variance', 'standard deviation', 'mean absolute deviation',
            'min', '5%', '25% (Q1)', '50% (median)', '75% (Q3)', '95%', 'max',
            'range = max − min', 'IQR = Q3 - Q1', 'CV = mean/std', 'QCD = (Q3-Q1)/(Q3+Q1)',
            'skew (As)', 'kurtosis (Es)'),
        'evaluation': (
            N, X_mean, Me, Mo,
            D_X, X_std, mad_X,
            min_X,
            np.percentile(X, 5), np.percentile(X, 25), np.percentile(X, 50), np.percentile(X, 75), np.percentile(X, 95),
            max_X,
            R, IQR, CV, QCD,
            As, Es),
        'conf.int.low': (
            '', conf_int_low_X_mean, conf_int_low_Me, '',

            conf_int_low_D_X,  conf_int_low_X_std, '',
            '', '', '', '', '', '', '',
            '', '', conf_int_low_CV, '',
            '', ''),
        'conf.int.high': (
            '', conf_int_up_X_mean, conf_int_up_Me, '',
            conf_int_up_D_X, conf_int_up_X_std,
            '', '', '', '', '', '', '', '',
            '', '', conf_int_up_CV, '',
            '', ''),
        'abs.err.': (
            '', abs_err_X_mean, abs_err_Me, '',
            abs_err_D_X, abs_err_X_std,
            '', '', '', '', '', '', '', '',
            '', '', abs_err_CV, '',
            abs_err_As, abs_err_Es),
        'rel.err.(%)': (
            '', rel_err_X_mean, rel_err_Me, '',
            rel_err_D_X, rel_err_X_std,
            '', '', '', '', '', '', '', '',
            '', '', rel_err_CV, '',
            rel_err_As, rel_err_Es),
        'note': (
            '', '', Me_note, '', '', '', '', '', '', '', '', '', '', '', '', '', CV_note, '',
            As_note, Es_note)
        })
            
    # вспомогательная таблица
    result_auxiliary = pd.DataFrame({
        'characteristic': (
            'confidence probability',
            'significance level',
            'count',
            "quantile of the Student's distribution",
            'quantile of the normal distribution',
            'quantile of the chi-square distribution (low)',
            'quantile of the chi-square distribution (up)'),
        'designation, formula': (
            'p',
            'a = 1 - p',
            'n',
            't(1 - a/2, n-1)',
            'u(p)',
            'chi2((1+p)/2, n-1)',
            'chi2((1-p)/2, n-1)'),
        'evaluation': (
            p_level, a_level, N, t_p, u_p, chi2_p_low, chi2_p_up)}
        )
    
    # вывод результатов
    if auxiliary_table:
        return result, result_auxiliary
    else:
        return result
    
    

#==============================================================================
#               ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
#==============================================================================    
    
#------------------------------------------------------------------------------
#   Функция graph_scatterplot_sns
#------------------------------------------------------------------------------

def graph_scatterplot_sns(
    X, Y,
    Xmin=None, Xmax=None,
    Ymin=None, Ymax=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    s=50,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    X = np.array(X)
    Y = np.array(Y)
    
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01       
    
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    #data_df = data_XY_df
    sns.scatterplot(
        x=X, y=Y,
        label=label_legend,
        s=s,
        palette=['orange'], color=color,
        ax=axes)
    axes.set_xlim(Xmin, Xmax)
    axes.set_ylim(Ymin, Ymax)        
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    



#------------------------------------------------------------------------------
#   Функция graph_scatterplot_mpl_np
#------------------------------------------------------------------------------

def graph_plot_mpl_np(
    X, Y,
    Xmin_in=None, Xmax_in=None,
    Ymin_in=None, Ymax_in=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    sns.set_style("whitegrid")    # настройка цветовой гаммы
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
    
    axes.plot(
        X, Y,
        linestyle = "-",
        color=color,
        linewidth=3,
        label=label_legend)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
    
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return


    
#------------------------------------------------------------------------------
#   Функция graph_scatterplot_sns_np
#------------------------------------------------------------------------------

def graph_plot_sns_np(
    X, Y,
    Xmin_in=None, Xmax_in=None,
    Ymin_in=None, Ymax_in=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    sns.set_style("darkgrid")    # настройка цветовой гаммы
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
        
    sns.lineplot(
        x=X, y=Y,
        palette=['orange'], color=color,
        linewidth=3,
        legend=True,
        label=label_legend,
        ax=axes)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
    
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_scatterplot_3D_mpl
#------------------------------------------------------------------------------

def graph_scatterplot_3D_mpl(
    X, Y, Z,
    elev=-160, azim=-60,
    Xmin_in=0, Xmax_in=3000,
    Ymin_in=-25, Ymax_in=25,
    Zmin_in=0, Zmax_in=20,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    z_label=None,
    label_fontsize=11, tick_fontsize=10,
    label_legend='', label_legend_fontsize=12,
    s=50,
    graph_size=(420/INCH, 297/INCH),
    file_name=None):
    
    sns.set_style("darkgrid")    # настройка цветовой гаммы
    fig = plt.figure(figsize=graph_size)
    axes = fig.add_subplot(111, projection='3d')
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
    
    if (Zmin_in != None) and (Zmax_in != None):
        Zmin=Zmin_in
        Zmax=Zmax_in
        axes.set_zlim(Zmin, Zmax)  
    
    axes.scatter(
        X, Y, Z,
        label=label_legend,
        s=50,
        color=color)
    
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.set_zlabel(z_label, fontsize = label_fontsize)
    axes.view_init(elev=elev, azim=azim)   
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return   



#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_mpl
#
#   Принимает в качестве аргумента np.ndarray и строит совмещенный график - 
#   гистограмму и коробчатую диаграмму на одном рисунке (Figure) - с помощью 
#    библиотеки Mathplotlib.
#------------------------------------------------------------------------------    
    

def graph_hist_boxplot_mpl(
    X,
    Xmin=None, Xmax=None,
    bins_hist='auto',
    density_hist=False,
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    label_fontsize=14, tick_fontsize=12, label_legend_fontsize=10,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    X = np.array(X)
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
        
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size)
    ax1 = plt.subplot(2,1,1)    # для гистограммы
    ax2 = plt.subplot(2,1,2)    # для коробчатой диаграммы  
    
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # выбор вида гистограммы (density=True/False - плотность/абс.частота) и параметры по оси OY
    ymax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
    if density_hist:
        label_hist = "эмпирическая плотность распределения"
        ax1.set_ylabel('Относительная плотность', fontsize = label_fontsize)
        ax1.set_ylim(0, ymax*1.4)
    else:
        label_hist = "эмпирическая частота"
        ax1.set_ylabel('Абсолютная частота', fontsize = label_fontsize)
        ax1.set_ylim(0, ymax*1.4)

    # данные для графика плотности распределения
    nx = 100
    hx = (Xmax - Xmin)/(nx - 1)
    x1 = np.linspace(Xmin, Xmax, nx)
    #hx = 0.1; nx =  int(floor((Xmax - Xmin)/hx)+1)
    #x1 = np.linspace(Xmin, Xmax, nx)
    y1 = sps.norm.pdf(x1, X.mean(), X.std(ddof = 1))
    
    # рендеринг гистограммы
    if density_hist:
        ax1.hist(
            X,
            bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
            density=density_hist,
            histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
            orientation='vertical',   # 'vertical', 'horizontal'
            color = "#1f77b4",
            label=label_hist)
        ax1.plot(
            x1, y1,
            linestyle = "-",
            color = "r",
            linewidth = 2,
            label = 'теоретическая нормальная кривая')
    else:
        ax1.hist(
            X,
            bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
            density=density_hist,
            histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
            orientation='vertical',   # 'vertical', 'horizontal'
            color = "#1f77b4",
            label=label_hist)    
        k = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        y2 = y1*len(X)*(max(X)-min(X))/k
        ax1.plot(
            x1, y2,
            linestyle = "-",
            color = "r",
            linewidth = 2,
            label = 'теоретическая нормальная кривая')
    
    # Среднее значение, медиана, мода на графике
    ax1.axvline(
        X.mean(),
        color='magenta', label = 'среднее значение', linewidth = 2)
    ax1.axvline(
        np.median(X),
        color='orange', label = 'медиана', linewidth = 2)
    ax1.axvline(stat.mode(X),
        color='cyan', label = 'мода', linewidth = 2)
    
    ax1.set_xlim(Xmin, Xmax)
    ax1.tick_params(labelsize = tick_fontsize)
    ax1.grid(True)
    ax1.legend(fontsize = label_legend_fontsize)
    
    # рендеринг коробчатой диаграммы
    ax2.boxplot(
        X,
        vert=False,
        notch=False,
        widths=0.5,
        patch_artist=True)
    ax2.set_xlim(Xmin, Xmax)
    ax2.set_xlabel(x_label, fontsize = label_fontsize)
    ax2.tick_params(labelsize = tick_fontsize)
    ax2.grid(True)
        
    # отображение графика на экране и сохранение в файл
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_probplot_sns
#   библиотеки Mathplotlib.
#------------------------------------------------------------------------------

def graph_hist_boxplot_probplot_sns(
    data,
    data_min=None, data_max=None,
    graph_inclusion='hbp',
    bins_hist='auto',
    density_hist=False,
    type_probplot='pp',
    title_figure=None, title_figure_fontsize=16,
    title_axes=None, title_axes_fontsize=14,
    data_label=None,
    label_fontsize=13, tick_fontsize=10, label_legend_fontsize=10,
    graph_size=None,
    file_name=None):
    
    X = np.array(data)
    Xmin = data_min
    Xmax = data_max
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
        
    # определение формы и размеров области рисования (Axes)
    count_graph = len(graph_inclusion)    # число графиков
    
    ax_rows = count_graph    # размерность области рисования (Axes)
    ax_cols = 1
    
    # создание рисунка (Figure) и области рисования (Axes)
    graph_size_dict = {
        1: (210/INCH, 297/INCH/2),
        2: (210/INCH, 297/INCH/1.5),
        3: (210/INCH, 297/INCH)}
    
    if not(graph_size):
        graph_size = graph_size_dict[count_graph]
    
    fig = plt.figure(figsize=graph_size)
    
    if count_graph == 3:
        ax1 = plt.subplot(3,1,1)
        ax2 = plt.subplot(3,1,2)
        ax3 = plt.subplot(3,1,3)
    elif count_graph == 2:
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
    elif count_graph == 1:
        ax1 = plt.subplot(1,1,1)
       
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # гистограмма (hist)
    if 'h' in graph_inclusion:
        X_mean = X.mean()
        X_std = X.std(ddof = 1)
        # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        xnorm1 = sps.norm.pdf(x1, X_mean, X_std)
        kx = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        xnorm2 = xnorm1*len(X)*(max(X)-min(X))/kx
        # выбор вида гистограммы (density=True/False - плотность/абс.частота) и параметры по оси OY
        xmax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        ax1.set_ylim(0, xmax*1.4)
        if density_hist:
            label_hist = "эмпирическая плотность распределения"
            ax1.set_ylabel('Относительная плотность', fontsize = label_fontsize)
        else:
            label_hist = "эмпирическая частота"
            ax1.set_ylabel('Абсолютная частота', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax1.hist(
                X,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)
            ax1.plot(
                x1, xnorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        else:
            ax1.hist(
                X,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)    
            ax1.plot(
                x1, xnorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        ax1.axvline(
            X_mean,
            color='magenta', label = 'среднее значение', linewidth = 2)
        ax1.axvline(
            np.median(X),
            color='orange', label = 'медиана', linewidth = 2)
        ax1.axvline(stat.mode(X),
            color='cyan', label = 'мода', linewidth = 2)
        ax1.set_xlim(Xmin, Xmax)
        ax1.legend(fontsize = label_legend_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax1.set_xlabel(data_label, fontsize = label_fontsize)
            
    # коробчатая диаграмма
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=X,
            orient='h',
            width=0.3,
            #color = "#1f77b4",
            ax = ax1 if (graph_inclusion == 'b') or (graph_inclusion == 'bp') else ax2)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax1.set_xlim(Xmin, Xmax)
            ax1.set_xlabel(data_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
        else:
            ax2.set_xlim(Xmin, Xmax)
            ax2.set_xlabel(data_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
    
    # вероятностный график
    gofplot = sm.ProbPlot(
        X,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                #color = "blue",
                ax = ax1 if (graph_inclusion == 'p') else ax3 if (graph_inclusion == 'hbp') else ax2)
            boxplot_xlabel = 'Theoretical probabilities'
            boxplot_ylabel = 'Sample probabilities'
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                #color = "blue",
                ax = ax1 if (graph_inclusion == 'p') else ax3 if (graph_inclusion == 'hbp') else ax2)
            boxplot_xlabel = 'Theoretical quantilies'
            boxplot_ylabel = 'Sample quantilies'
        boxplot_legend = ['эмпирические данные', 'нормальное распределение']
        if (graph_inclusion == 'p'):
            ax1.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax1.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax1.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax1.tick_params(labelsize = tick_fontsize)
        elif (graph_inclusion == 'hbp'):
            ax3.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax3.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax3.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax3.tick_params(labelsize = tick_fontsize)
        else:
            ax2.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax2.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax2.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax2.tick_params(labelsize = tick_fontsize)
            
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    



#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_probplot_XY_sns
#   библиотеки Mathplotlib.
#------------------------------------------------------------------------------

def graph_hist_boxplot_probplot_XY_sns(
    data_X, data_Y,
    data_X_min=None, data_X_max=None,
    data_Y_min=None, data_Y_max=None,
    graph_inclusion='hbp',
    bins_hist='auto',
    density_hist=False,
    type_probplot='pp',
    title_figure=None, title_figure_fontsize=16,
    x_title='X', y_title='Y', title_axes_fontsize=14,
    data_X_label=None,
    data_Y_label=None,
    label_fontsize=13, tick_fontsize=10, label_legend_fontsize=10,
    graph_size=None,
    file_name=None):
    
    X = np.array(data_X)
    Xmin = data_X_min
    Xmax = data_X_max
    Y = np.array(data_Y)
    Ymin = data_Y_min
    Ymax = data_Y_max
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01        
        
    # определение формы и размеров области рисования (Axes)
    count_graph = len(graph_inclusion)    # число графиков
    
    ax_rows = count_graph    # размерность области рисования (Axes)
    ax_cols = 2
    
    # создание рисунка (Figure) и области рисования (Axes)
    graph_size_dict = {
        1: (420/INCH, 297/INCH/2),
        2: (420/INCH, 297/INCH/1.5),
        3: (420/INCH, 297/INCH)}
    
    if not(graph_size):
        graph_size = graph_size_dict[count_graph]
    
    fig = plt.figure(figsize=graph_size)
    
    if count_graph == 3:
        ax1 = plt.subplot(3,2,1)
        ax2 = plt.subplot(3,2,2)
        ax3 = plt.subplot(3,2,3)
        ax4 = plt.subplot(3,2,4)
        ax5 = plt.subplot(3,2,5)
        ax6 = plt.subplot(3,2,6)
    elif count_graph == 2:
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
    elif count_graph == 1:
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
       
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(x_title, fontsize = title_axes_fontsize)
    ax2.set_title(y_title, fontsize = title_axes_fontsize)
    
    # гистограмма (hist) X
    if 'h' in graph_inclusion:
        X_mean = X.mean()
        X_std = X.std(ddof = 1)
        # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        xnorm1 = sps.norm.pdf(x1, X_mean, X_std)
        kx = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        xnorm2 = xnorm1*len(X)*(max(X)-min(X))/kx
        # выбор вида гистограммы (density=True/False - плотность/абс.частота) и параметры по оси OY
        xmax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        ax1.set_ylim(0, xmax*1.4)
        if density_hist:
            label_hist = "эмпирическая плотность распределения"
            ax1.set_ylabel('Относительная плотность', fontsize = label_fontsize)
        else:
            label_hist = "эмпирическая частота"
            ax1.set_ylabel('Абсолютная частота', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax1.hist(
                X,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)
            ax1.plot(
                x1, xnorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        else:
            ax1.hist(
                X,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)    
            ax1.plot(
                x1, xnorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        ax1.axvline(
            X_mean,
            color='magenta', label = 'среднее значение', linewidth = 2)
        ax1.axvline(
            np.median(X),
            color='orange', label = 'медиана', linewidth = 2)
        ax1.axvline(stat.mode(X),
            color='cyan', label = 'мода', linewidth = 2)
        ax1.set_xlim(Xmin, Xmax)
        ax1.legend(fontsize = label_legend_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)

# гистограмма (hist) Y
    if 'h' in graph_inclusion:
        Y_mean = Y.mean()
        Y_std = Y.std(ddof = 1)
        # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        ny = 100
        hy = (Ymax - Ymin)/(ny - 1)
        y1 = np.linspace(Ymin, Ymax, ny)
        ynorm1 = sps.norm.pdf(y1, Y_mean, Y_std)
        ky = len(np.histogram(Y, bins=bins_hist, density=density_hist)[0])
        ynorm2 = ynorm1*len(Y)*(max(Y)-min(Y))/ky
        # выбор вида гистограммы (density=True/False - плотность/абс.частота) и параметры по оси OY
        ymax = max(np.histogram(Y, bins=bins_hist, density=density_hist)[0])
        ax2.set_ylim(0, ymax*1.4)
        if density_hist:
            label_hist = "эмпирическая плотность распределения"
            ax2.set_ylabel('Относительная плотность', fontsize = label_fontsize)
        else:
            label_hist = "эмпирическая частота"
            ax2.set_ylabel('Абсолютная частота', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax2.hist(
                Y,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)
            ax2.plot(
                y1, ynorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        else:
            ax2.hist(
                Y,
                bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',   # 'vertical', 'horizontal'
                color = "#1f77b4",
                label=label_hist)    
            ax2.plot(
                y1, ynorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'теоретическая нормальная кривая')
        ax2.axvline(
            Y_mean,
            color='magenta', label = 'среднее значение', linewidth = 2)
        ax2.axvline(
            np.median(Y),
            color='orange', label = 'медиана', linewidth = 2)
        ax2.axvline(stat.mode(Y),
            color='cyan', label = 'мода', linewidth = 2)
        ax2.set_xlim(Ymin, Ymax)
        ax2.legend(fontsize = label_legend_fontsize)
        ax2.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
        
    # коробчатая диаграмма X
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=X,
            orient='h',
            width=0.3,
            ax = ax1 if (graph_inclusion == 'b') or (graph_inclusion == 'bp') else ax3)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax1.set_xlim(Xmin, Xmax)
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)
        else:
            ax3.set_xlim(Xmin, Xmax)
            ax3.set_xlabel(data_X_label, fontsize = label_fontsize)
            ax3.tick_params(labelsize = tick_fontsize)
            
    # коробчатая диаграмма Y
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=Y,
            orient='h',
            width=0.3,
            ax = ax2 if (graph_inclusion == 'b') or (graph_inclusion == 'bp') else ax4)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax2.set_xlim(Ymin, Ymax)
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
        else:
            ax4.set_xlim(Ymin, Ymax)
            ax4.set_xlabel(data_Y_label, fontsize = label_fontsize)
            ax4.tick_params(labelsize = tick_fontsize)            
    
    # вероятностный график X
    gofplot = sm.ProbPlot(
        X,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                ax = ax1 if (graph_inclusion == 'p') else ax5 if (graph_inclusion == 'hbp') else ax3)
            boxplot_xlabel = 'Theoretical probabilities'
            boxplot_ylabel = 'Sample probabilities'
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                ax = ax1 if (graph_inclusion == 'p') else ax5 if (graph_inclusion == 'hbp') else ax3)
            boxplot_xlabel = 'Theoretical quantilies'
            boxplot_ylabel = 'Sample quantilies'
        boxplot_legend = ['эмпирические данные', 'нормальное распределение']
        if (graph_inclusion == 'p'):
            ax1.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax1.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax1.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax1.tick_params(labelsize = tick_fontsize)
        elif (graph_inclusion == 'hbp'):
            ax5.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax5.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax5.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax5.tick_params(labelsize = tick_fontsize)
        else:
            ax3.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax3.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax3.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax3.tick_params(labelsize = tick_fontsize)
            
    # вероятностный график Y
    gofplot = sm.ProbPlot(
        Y,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                ax = ax2 if (graph_inclusion == 'p') else ax6 if (graph_inclusion == 'hbp') else ax4)
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                ax = ax2 if (graph_inclusion == 'p') else ax6 if (graph_inclusion == 'hbp') else ax4)
        boxplot_legend = ['эмпирические данные', 'нормальное распределение']
        if (graph_inclusion == 'p'):
            ax2.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax2.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax2.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax2.tick_params(labelsize = tick_fontsize)
        elif (graph_inclusion == 'hbp'):
            ax6.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax6.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax6.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax6.tick_params(labelsize = tick_fontsize)
        else:
            ax4.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax4.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax4.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax4.tick_params(labelsize = tick_fontsize)            
            
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    



#------------------------------------------------------------------------------
#   Функция graph_ecdf_cdf_mpl
#------------------------------------------------------------------------------

def graph_ecdf_cdf_mpl(
    X,
    Xmin=None, Xmax=None,
    title_figure=None,
    title_axes=None,
    x_label=None,
    graph_size=(210/INCH, 297/INCH/2),
    file_name=None):
    
    X = np.array(X)
    
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size)
    ax = plt.subplot(2,1,1)    # для гистограммы
        
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = f_size+6)
    
    # заголовок области рисования (Axes)
    ax.set_title(title_axes)
    
    # эмпирическая функция распределения
    from statsmodels.distributions.empirical_distribution import ECDF
    x1 = np.array(X)
    ecdf = ECDF(x1)
    x1.sort()
    y1 = ecdf(x1)
        
    # теоретическая функция распределения
    hx = 0.1; nx = floor((Xmax - Xmin)/hx)+1
    x2 = np.linspace(Xmin, Xmax, nx)
    y2 = sps.norm.cdf(x2, X.mean(), X.std(ddof = 1))
    
    # рендеринг эмпирической функции распределения
    ax.step(x1, y1,
            where='post',
            label = 'эмпирическая функция распределения')
    
    # рендеринг теоретической функции распределения
    ax.plot(
       x2, y2,
       linestyle = "-",
       color = "r",
       linewidth = 2,
       label = 'теоретическая функция нормального распределения')
    
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(x_label)
    ax.grid(True)
    ax.legend(fontsize = 12)
    
    # отображение графика на экране и сохранение в файл
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    


#------------------------------------------------------------------------------
#   Функция graph_regression_plot_sns
#------------------------------------------------------------------------------

def graph_regression_plot_sns(
    X, Y,
    regression_model,
    Xmin=None, Xmax=None,
    Ymin=None, Ymax=None,
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12, 
    label_legend_regr_model='', label_legend_fontsize=12,
    s=50, linewidth_regr_model=2,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    X = np.array(X)
    Y = np.array(Y)
    
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01       
    
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # фактические данные
    sns.scatterplot(
        x=X, y=Y,
        label='фактические данные',
        s=s,
        color='red',
        ax=axes)
    
    # график регрессионной модели
    nx = 100
    hx = (Xmax - Xmin)/(nx - 1)
    x1 = np.linspace(Xmin, Xmax, nx)
    y1 = regression_model(x1)
    sns.lineplot(
        x=x1, y=y1,
        color='blue',
        linewidth=linewidth_regr_model,
        legend=True,
        label=label_legend_regr_model,
        ax=axes)
    
    # график остатков
    
    axes.set_xlim(Xmin, Xmax)
    axes.set_ylim(Ymin, Ymax)        
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    




#==============================================================================
#               ФУНКЦИИ ДЛЯ ВЫЯВЛЕНИЯ АНОМАЛЬНЫХ ЗНАЧЕНИЙ (ВЫБРОСОВ)
#==============================================================================    
    
#------------------------------------------------------------------------------
#   Функция detecting_outliers_mad_test
#   Критерий максимального абсолютного отклонения (М.А.О.) (см.[Кобзарь, с.547]
#------------------------------------------------------------------------------

def detecting_outliers_mad_test(
        X_in,
        p_level=0.95,
        show_in_full=False,
        delete_outlier=False):
    N = len(X_in)
    X_mean = X_in.mean()
    X_std = X_in.std(ddof = 1)
    # создаем вспомогательный DataFrame, отсортированный по возрастанию
    data = pd.DataFrame({'value': X_in}).sort_values(by='value')
    # расчетное значение статистики критерия М.А.О.
    data['mad_calc'] = abs((data['value'] - X_mean)/X_std)
    # аппроксимация табличного значения статистики критерия М.А.О.
    mad_table = lambda n: 1.39 + 0.462*log(n-3) if 5<=n<35 else 2.136 - 0.281*log(n-15) if 35<=n<=500 else '-'
    data['mad_table'] = mad_table(N)
    # добавляем во вспомогательный DataFrame столбец 'outlier_conclusion', в котором выбросу будут отмечены как 'outlier'
    data['outlier_conclusion'] = '-'
    mask = data['mad_calc'] >= data['mad_table']
    data.loc[mask, 'outlier_conclusion'] = 'outlier'
        
    if delete_outlier:
        X_out = np.delete(X_in, [data.loc[mask].index])
        return data, X_out
    else:
        return data




#==============================================================================
#               3. ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЯ ЗАКОНОВ РАСПРЕДЕЛЕНИЯ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция norm_distr_check
#------------------------------------------------------------------------------


def norm_distr_check (data, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(data)
    N = len(X)
        
    # критерий Шапиро-Уилка
    if N >= 8:
        result_ShW = sci.stats.shapiro(X)
        s_calc_ShW = result_ShW.statistic
        a_calc_ShW = result_ShW.pvalue
        conclusion_ShW = 'gaussian distribution' if a_calc_ShW >= a_level else 'not gaussian distribution'
    else:
        result_ShW = '-'
        s_calc_ShW = '-'
        a_calc_ShW = '-'
        conclusion_ShW = 'count less than 8'

    # критерий Эппса-Палли        
    сdf_beta_I = lambda x, a, b: sci.stats.beta.cdf(x, a, b, loc=0, scale=1)
    g_beta_III = lambda z, δ: δ*z / (1+(δ-1)*z)
    cdf_beta_III = lambda x, θ0, θ1, θ2, θ3, θ4: сdf_beta_I(g_beta_III((x - θ4)/θ3, θ2), θ0, θ1)
    
    θ_1 = (1.8645, 2.5155, 5.8256, 0.9216, 0.0008)    # для 15 < n < 50
    θ_2 = (1.7669, 2.1668, 6.7594, 0.91, 0.0016)    # для n >= 50
    
    if N >= 8:
        X_mean = X.mean()
        m2 = np.var(X, ddof = 0)
        # расчетное значение статистики критерия
        A = sqrt(2) * np.sum([exp(-(X[i] - X_mean)**2 / (4*m2)) for i in range(N)])
        B = 2/N * np.sum(
            [np.sum([exp(-(X[j] - X[k])**2 / (2*m2)) for j in range(0, k)]) 
             for k in range(1, N)])
        s_calc_EP = 1 + N / sqrt(3) + B - A
        # табличное значение статистики критерия
        Tep_table_df = pd.read_csv(
            filepath_or_buffer='table/Tep_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.9:   Tep_table_df.columns[0],
            0.95:  Tep_table_df.columns[1],
            0.975: Tep_table_df.columns[2],
            0.99:  Tep_table_df.columns[3]}
        f_lin = sci.interpolate.interp1d(Tep_table_df.index, Tep_table_df[p_level_dict[p_level]])
        critical_value_EP = float(f_lin(N))
        # проверка гипотезы
        if 15 < N < 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_1[0], θ_1[1], θ_1[2], θ_1[3], θ_1[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        elif N >= 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_2[0], θ_2[1], θ_2[2], θ_2[3], θ_2[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        else:
            a_calc_EP = ''              
            conclusion_EP = 'gaussian distribution' if s_calc_EP <= critical_value_EP else 'not gaussian distribution'            
                
    else:
        s_calc_EP = '-'
        critical_value_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count less than 8'
    
    
    # критерий Д'Агостино (K2-test)
    if N >= 8:
        result_K2 = sci.stats.normaltest(X)
        s_calc_K2 = result_K2.statistic
        a_calc_K2 = result_K2.pvalue
        conclusion_K2 = 'gaussian distribution' if a_calc_K2 >= a_level else 'not gaussian distribution'
    else:
        s_calc_K2 = '-'
        a_calc_K2 = '-'
        conclusion_K2 = 'count less than 8'
    
    # критерий Андерсона-Дарлинга
    result_AD = sci.stats.anderson(X)
    df_AD = pd.DataFrame({
        'a_level (%)': result_AD.significance_level,
        'statistic': [result_AD.statistic for i in range(len(result_AD.critical_values))],
        'critical_value': result_AD.critical_values
        })
    statistic_AD = float(df_AD[df_AD['a_level (%)'] == round((1 - p_level)*100, 1)]['statistic'])
    critical_value_AD = float(df_AD[df_AD['a_level (%)'] == round((1 - p_level)*100, 1)]['critical_value'])
    conclusion_AD = 'gaussian distribution' if statistic_AD < critical_value_AD else 'not gaussian distribution'
    
    # критерий Колмогорова-Смирнова
    if N >= 50:
        result_KS = sci.stats.kstest(X, 'norm')
        s_calc_KS = result_KS.statistic
        a_calc_KS = result_KS.pvalue
        conclusion_KS = 'gaussian distribution' if a_calc_KS >= a_level else 'not gaussian distribution'
    else:
        s_calc_KS = '-'
        a_calc_KS = '-'
        conclusion_KS = 'count less than 50'
        
    # критерий Лиллиефорса      
    from statsmodels.stats.diagnostic import lilliefors
    s_calc_L, a_calc_L = sm.stats.diagnostic.lilliefors(X, 'norm')
    conclusion_L = 'gaussian distribution' if a_calc_L >= a_level else 'not gaussian distribution'  
    
    # критерий Крамера-Мизеса-Смирнова (омега-квадрат)
    if N >= 40:
        result_CvM = sci.stats.cramervonmises(X, 'norm')
        s_calc_CvM = result_CvM.statistic
        a_calc_CvM = result_CvM.pvalue
        conclusion_CvM = 'gaussian distribution' if a_calc_CvM >= a_level else 'not gaussian distribution'
    else:
        s_calc_CvM = '-'
        a_calc_CvM = '-'
        conclusion_CvM = 'count less than 40'
    
    # критерий Пирсона (хи-квадрат)
    if N >= 100:
        ddof = 2    # поправка к числу степеней свободы (число параметров распределения, оцениваемое по выборке)
        result_Chi2 = sci.stats.chisquare(X, ddof=ddof)
        s_calc_Chi2 = result_Chi2.statistic
        a_calc_Chi2 = result_Chi2.pvalue
        conclusion_Chi2 = 'gaussian distribution' if a_calc_Chi2 >= a_level else 'not gaussian distribution'
    else:
        s_calc_Chi2 = '-'
        a_calc_Chi2 = '-'
        conclusion_Chi2 = 'count less than 100'
        
    # критерий Харке-Бера (асимметрии и эксцесса)
    if N > 2000:
        result_JB = sci.stats.jarque_bera(X)
        s_calc_JB = result_JB.statistic
        a_calc_JB = result_JB.pvalue
        conclusion_JB = 'gaussian distribution' if a_calc_JB >= a_level else 'not gaussian distribution'
    else:
        s_calc_JB = '-'
        a_calc_JB = '-'
        conclusion_JB = 'count less than 2000'
    
    # критерий асимметрии
    if N >= 8:
        result_As = sci.stats.skewtest(X)
        s_calc_As = result_As.statistic
        a_calc_As = result_As.pvalue
        conclusion_As = 'gaussian distribution' if a_calc_As >= a_level else 'not gaussian distribution'
    else:
        s_calc_As = '-'
        a_calc_As = '-'
        conclusion_As = 'count less than 8'
     
    # критерий эксцесса
    if N > 20:
        result_Es = sci.stats.kurtosistest(X)
        s_calc_Es = result_Es.statistic
        a_calc_Es = result_Es.pvalue
        conclusion_Es = 'gaussian distribution' if a_calc_Es >= a_level else 'not gaussian distribution'
    else:
        s_calc_Es = '-'
        a_calc_Es = '-'
        conclusion_Es = 'count less than 20'
    
    # Создадим DataFrame для сводки результатов
    #------------------------------------------
    result = pd.DataFrame({
    'test': (
        'Shapiro-Wilk test',
        'Epps-Pulley test',
        "D'Agostino's K-squared test",
        'Anderson-Darling test',
        'Kolmogorov–Smirnov test',
        'Lilliefors test',
        'Cramér–von Mises test',
        'Chi-squared test',
        'Jarque–Bera test',
        'skewtest',
        'kurtosistest'),
    'p_level': (p_level),
    'a_level': (a_level),
    'a_calc': (
        a_calc_ShW,
        a_calc_EP,
        a_calc_K2,
        '',
        a_calc_KS,
        a_calc_L,
        a_calc_CvM,
        a_calc_Chi2,
        a_calc_JB,
        a_calc_As,
        a_calc_Es),
    'a_calc >= a_level': (
        a_calc_ShW >= a_level if N >= 8 else '-',
        a_calc_EP >= a_level if N > 15 else '-',
        a_calc_K2 >= a_level if N >= 8 else '-',
        '',
        a_calc_KS >= a_level if N >= 50 else '-',
        a_calc_L >= a_level,
        a_calc_CvM >= a_level if N >= 40 else '-',
        a_calc_Chi2 >= a_level if N >= 100 else '-',
        a_calc_JB >= a_level if N >= 2000 else '-',
        a_calc_As >= a_level if N >= 8 else '-',
        a_calc_Es >= a_level if N > 20 else '-'),
    'statistic': (
        s_calc_ShW,
        s_calc_EP,
        s_calc_K2,
        statistic_AD,
        s_calc_KS,
        s_calc_L,
        s_calc_CvM,
        s_calc_Chi2,
        s_calc_JB,
        s_calc_As,
        s_calc_Es),
    'critical_value': (
        '',
        critical_value_EP,
        '',
        critical_value_AD,
        '', '', '', '', '', '', ''),
    'statistic < critical_value': (
        '',
        s_calc_EP < critical_value_EP  if N >= 8 else '-',
        '',
        statistic_AD < critical_value_AD,
        '', '', '', '', '', '', ''),
    'conclusion': (
        conclusion_ShW,
        conclusion_EP,
        conclusion_K2,
        conclusion_AD,
        conclusion_KS,
        conclusion_L,
        conclusion_CvM,
        conclusion_Chi2,
        conclusion_JB,
        conclusion_As,
        conclusion_Es
        )})  
        
    return result


#------------------------------------------------------------------------------
#   Функция Shapiro_Wilk_test
#   функция для обработки реализации теста Шапиро-Уилка
#------------------------------------------------------------------------------

def Shapiro_Wilk_test(data, p_level=0.95, DecPlace=5):
    a_level = 1 - p_level
    data = np.array(data)
    result = sci.stats.shapiro(data)
    s_calc = result.statistic    # расчетное значение статистики критерия
    a_calc = result.pvalue    # расчетный уровень значимости
    
    print(f"Расчетный уровень значимости: a_calc = {round(a_calc, DecPlace)}")
    print(f"Заданный уровень значимости: a_level = {round(a_level, DecPlace)}")
    
    if a_calc >= a_level:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} >= a_level = {round(a_level, DecPlace)}" + \
            ", то гипотеза о нормальности распределения по критерию Шапиро-Уилка ПРИНИМАЕТСЯ"
    else:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} < a_level = {round(a_level, DecPlace)}" + \
            ", то гипотеза о нормальности распределения по критерию Шапиро-Уилка ОТВЕРГАЕТСЯ"
    print(conclusion_ShW_test)



#------------------------------------------------------------------------------
#   Функция Epps_Pulley_test
#------------------------------------------------------------------------------    

def Epps_Pulley_test(data, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(data)
    N = len(X)
    
    # аппроксимация предельных распределений статистики критерия
    сdf_beta_I = lambda x, a, b: sci.stats.beta.cdf(x, a, b, loc=0, scale=1)
    g_beta_III = lambda z, δ: δ*z / (1+(δ-1)*z)
    cdf_beta_III = lambda x, θ0, θ1, θ2, θ3, θ4: сdf_beta_I(g_beta_III((x - θ4)/θ3, θ2), θ0, θ1)
    # набор параметров распределения
    θ_1 = (1.8645, 2.5155, 5.8256, 0.9216, 0.0008)    # для 15 < n < 50
    θ_2 = (1.7669, 2.1668, 6.7594, 0.91, 0.0016)    # для n >= 50
    
    if N >= 8:
        # среднее арифметическое
        X_mean = X.mean()
        # центральный момент 2-го порядка
        m2 = np.var(X, ddof = 0)
        # расчетное значение статистики критерия
        A = sqrt(2) * np.sum([exp(-(X[i] - X_mean)**2 / (4*m2)) for i in range(N)])
        B = 2/N * np.sum(
            [np.sum([exp(-(X[j] - X[k])**2 / (2*m2)) for j in range(0, k)]) 
             for k in range(1, N)])
        s_calc_EP = 1 + N / sqrt(3) + B - A
        # табличное значение статистики критерия
        Tep_table_df = pd.read_csv(
            filepath_or_buffer='table/Tep_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.9:   Tep_table_df.columns[0],
            0.95:  Tep_table_df.columns[1],
            0.975: Tep_table_df.columns[2],
            0.99:  Tep_table_df.columns[3]}
        f_lin = sci.interpolate.interp1d(Tep_table_df.index, Tep_table_df[p_level_dict[p_level]])
        s_table_EP = float(f_lin(N))
        # проверка гипотезы
        if 15 < N < 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_1[0], θ_1[1], θ_1[2], θ_1[3], θ_1[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        elif N >= 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_2[0], θ_2[1], θ_2[2], θ_2[3], θ_2[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        else:
            a_calc_EP = ''              
            conclusion_EP = 'gaussian distribution' if s_calc_EP <= s_table_EP else 'not gaussian distribution'            
                
    else:
        s_calc_EP = '-'
        s_table_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count less than 8'
    
    result = pd.DataFrame({
        'test': ('Epps-Pulley test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'a_calc': (a_calc_EP),
        'a_calc >= a_level': (a_calc_EP >= a_level if N > 15 else '-'),
        'statistic': (s_calc_EP),
        'critical_value': (s_table_EP),
        'statistic < critical_value': (s_calc_EP < s_table_EP if N >= 8 else '-'),
        'conclusion': (conclusion_EP)},
        index=[1])  
        
    return result



#==============================================================================
#               ФУНКЦИИ ДЛЯ КОРРЕЛЯЦИОННОГО АНАЛИЗА
#==============================================================================


#------------------------------------------------------------------------------
#   Функция Cheddock_scale_check
#------------------------------------------------------------------------------

def Cheddock_scale_check(r, name='r'):
    # задаем шкалу Чеддока
    Cheddock_scale = {
        f'no correlation (|{name}| <= 0.1)':    0.1,
        f'very weak (0.1 < |{name}| <= 0.2)':   0.2,
        f'weak (0.2 < |{name}| <= 0.3)':        0.3,
        f'moderate (0.3 < |{name}| <= 0.5)':    0.5,
        f'perceptible (0.5 < |{name}| <= 0.7)': 0.7,
        f'high (0.7 < |{name}| <= 0.9)':        0.9,
        f'very high (0.9 < |{name}| <= 0.99)':  0.99,
        f'functional (|{name}| > 0.99)':        1.0}
    
    r_scale = list(Cheddock_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Cheddock_scale = list(Cheddock_scale.keys())[i]
            break
    return conclusion_Cheddock_scale  



#------------------------------------------------------------------------------
#   Функция Evans_scale_check
#------------------------------------------------------------------------------

def Evans_scale_check(r, name='r'):
    # задаем шкалу Эванса
    Evans_scale = {
        f'very weak (|{name}| < 0.19)':         0.2,
        f'weak (0.2 < |{name}| <= 0.39)':       0.4,  
        f'moderate (0.4 < |{name}| <= 0.59)':   0.6,
        f'strong (0.6 < |{name}| <= 0.79)':     0.8,
        f'very strong (0.8 < |{name}| <= 1.0)': 1.0}
    
    r_scale = list(Evans_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Evans_scale = list(Evans_scale.keys())[i]
            break
    return conclusion_Evans_scale  



#------------------------------------------------------------------------------
#   Функция corr_coef_check
#------------------------------------------------------------------------------

# Функция для расчета и анализа коэффициента корреляции Пирсона

def corr_coef_check(X, Y, p_level=0.95, scale='Cheddok'):
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)
    # оценка коэффициента линейной корреляции средствами scipy
    corr_coef, a_corr_coef_calc = sci.stats.pearsonr(X, Y)
    # несмещенная оценка коэффициента линейной корреляции (при n < 15) (см.Кобзарь, с.607)
    if n_X < 15:
        corr_coef = corr_coef * (1 + (1 - corr_coef**2) / (2*(n_X-3)))
    # проверка гипотезы о значимости коэффициента корреляции
    t_corr_coef_calc = abs(corr_coef) * sqrt(n_X-2) / sqrt(1 - corr_coef**2)
    t_corr_coef_table = sci.stats.t.ppf((1 + p_level)/2 , n_X - 2)
    conclusion_corr_coef_sign = 'significance' if t_corr_coef_calc >= t_corr_coef_table else 'not significance'
    # доверительный интервал коэффициента корреляции
    if t_corr_coef_calc >= t_corr_coef_table:
        z1 = np.arctanh(corr_coef) - sci.stats.norm.ppf((1 + p_level)/2, 0, 1) / sqrt(n_X-3) - corr_coef / (2*(n_X-1))
        z2 = np.arctanh(corr_coef) + sci.stats.norm.ppf((1 + p_level)/2, 0, 1) / sqrt(n_X-3) - corr_coef / (2*(n_X-1))
        corr_coef_conf_int_low = tanh(z1)
        corr_coef_conf_int_high = tanh(z2)
    else:
        corr_coef_conf_int_low = corr_coef_conf_int_high = '-'    
    # оценка тесноты связи
    if scale=='Cheddok':
        conclusion_corr_coef_scale = scale + ': ' + Cheddock_scale_check(corr_coef)
    elif scale=='Evans':
        conclusion_corr_coef_scale = scale + ': ' + Evans_scale_check(corr_coef)
    # формируем результат            
    result = pd.DataFrame({
        'notation': ('r'),
        'coef_value': (corr_coef),
        'coef_value_squared': (corr_coef**2),
        'p_level': (p_level),
        'a_level': (a_level),
        't_calc': (t_corr_coef_calc),
        't_table': (t_corr_coef_table),
        't_calc >= t_table': (t_corr_coef_calc >= t_corr_coef_table),
        'a_calc': (a_corr_coef_calc),
        'a_calc <= a_level': (a_corr_coef_calc <= a_level),
        'significance_check': (conclusion_corr_coef_sign),
        'conf_int_low': (corr_coef_conf_int_low),
        'conf_int_high': (corr_coef_conf_int_high),
        'scale': (conclusion_corr_coef_scale)
        },
        index=['Correlation coef.'])
        
    return result        


#------------------------------------------------------------------------------
#   Функция corr_ratio_check
#------------------------------------------------------------------------------

# Функция для расчета и анализа корреляционного отношения

def corr_ratio_check(X, Y, p_level=0.95, orientation='XY', scale='Cheddok'):
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)    
    # запишем данные в DataFrame
    matrix_XY_df = pd.DataFrame({
        'X': X,
        'Y': Y})
    # число интервалов группировки
    group_int_number = lambda n: round (3.31*log(n, 10)+1) if round (3.31*log(n, 10)+1) >=2 else 2
    K_X = group_int_number(n_X)
    K_Y = group_int_number(n_Y)
    # группировка данных и формирование корреляционной таблицы
    cut_X = pd.cut(X, bins=K_X)
    cut_Y = pd.cut(Y, bins=K_Y)
    matrix_XY_df['cut_X'] = cut_X
    matrix_XY_df['cut_Y'] = cut_Y
    CorrTable_df = pd.crosstab(
        index=matrix_XY_df['cut_X'],
        columns=matrix_XY_df['cut_Y'],
        rownames=['cut_X'],
        colnames=['cut_Y'],
        dropna=False)
    CorrTable_np = np.array(CorrTable_df)
    # итоги корреляционной таблицы по строкам и столбцам
    n_group_X = [np.sum(CorrTable_np[i]) for i in range(K_X)]
    n_group_Y = [np.sum(CorrTable_np[:,j]) for j in range(K_Y)]
    # среднегрупповые значения переменной X
    Xboun_mean = [(CorrTable_df.index[i].left + CorrTable_df.index[i].right)/2 for i in range(K_X)]
    Xboun_mean[0] = (np.min(X) + CorrTable_df.index[0].right)/2    # исправляем значения в крайних интервалах
    Xboun_mean[K_X-1] = (CorrTable_df.index[K_X-1].left + np.max(X))/2
    # среднегрупповые значения переменной Y
    Yboun_mean = [(CorrTable_df.columns[j].left + CorrTable_df.columns[j].right)/2 for j in range(K_Y)]
    Yboun_mean[0] = (np.min(Y) + CorrTable_df.columns[0].right)/2    # исправляем значения в крайних интервалах
    Yboun_mean[K_Y-1] = (CorrTable_df.columns[K_Y-1].left + np.max(Y))/2
    # средневзевешенные значения X и Y для каждой группы
    Xmean_group = [np.sum(CorrTable_np[:,j] * Xboun_mean) / n_group_Y[j] for j in range(K_Y)]
    for i, elem in enumerate(Xmean_group):
        if isnan(elem):
            Xmean_group[i] = 0
    Ymean_group = [np.sum(CorrTable_np[i] * Yboun_mean) / n_group_X[i] for i in range(K_X)]
    for i, elem in enumerate(Ymean_group):
        if isnan(elem):
            Ymean_group[i] = 0
    # общая дисперсия X и Y
    Sum2_total_X = np.sum(n_group_X * (Xboun_mean - np.mean(X))**2)
    Sum2_total_Y = np.sum(n_group_Y * (Yboun_mean - np.mean(Y))**2)
    # межгрупповая дисперсия X и Y (дисперсия групповых средних)
    Sum2_between_group_X = np.sum(n_group_Y * (Xmean_group - np.mean(X))**2)
    Sum2_between_group_Y = np.sum(n_group_X * (Ymean_group - np.mean(Y))**2)
    # эмпирическое корреляционное отношение
    corr_ratio_XY = sqrt(Sum2_between_group_Y / Sum2_total_Y)
    corr_ratio_YX = sqrt(Sum2_between_group_X / Sum2_total_X)
    try:
        if orientation!='XY' and orientation!='YX':
            raise ValueError("Error! Incorrect orientation!")
        if orientation=='XY':
            corr_ratio = corr_ratio_XY
        elif orientation=='YX':
            corr_ratio = corr_ratio_YX
    except ValueError as err:
        print(err)
    # проверка гипотезы о значимости корреляционного отношения
    F_corr_ratio_calc = (n_X - K_X)/(K_X - 1) * corr_ratio**2 / (1 - corr_ratio**2)
    dfn = K_X - 1
    dfd = n_X - K_X
    F_corr_ratio_table = sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
    a_corr_ratio_calc = 1 - sci.stats.f.cdf(F_corr_ratio_calc, dfn, dfd, loc=0, scale=1)
    conclusion_corr_ratio_sign = 'significance' if F_corr_ratio_calc >= F_corr_ratio_table else 'not significance'
    # доверительный интервал корреляционного отношения
    if F_corr_ratio_calc >= F_corr_ratio_table:
        f1 = round ((K_X - 1 + n_X * corr_ratio**2)**2 / (K_X - 1 + 2 * n_X * corr_ratio**2))
        f2 = n_X - K_X
        z1 = (n_X - K_X) / n_X * corr_ratio**2 / (1 - corr_ratio**2) * 1/sci.stats.f.ppf(p_level, f1, f2, loc=0, scale=1) - (K_X - 1)/n_X
        z2 = (n_X - K_X) / n_X * corr_ratio**2 / (1 - corr_ratio**2) * 1/sci.stats.f.ppf(1 - p_level, f1, f2, loc=0, scale=1) - (K_X - 1)/n_X
        corr_ratio_conf_int_low = sqrt(z1) if sqrt(z1) >= 0 else 0
        corr_ratio_conf_int_high = sqrt(z2) if sqrt(z2) <= 1 else 1
    else:
        corr_ratio_conf_int_low = corr_ratio_conf_int_high = '-'    
    # оценка тесноты связи
    if scale=='Cheddok':
        conclusion_corr_ratio_scale = scale + ': ' + Cheddock_scale_check(corr_ratio, name=chr(951))
    elif scale=='Evans':
        conclusion_corr_ratio_scale = scale + ': ' + Evans_scale_check(corr_ratio, name=chr(951))
    # формируем результат            
    result = pd.DataFrame({
        'notation': (chr(951)),
        'coef_value': (corr_ratio),
        'coef_value_squared': (corr_ratio**2),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_corr_ratio_calc),
        'F_table': (F_corr_ratio_table),
        'F_calc >= F_table': (F_corr_ratio_calc >= F_corr_ratio_table),
        'a_calc': (a_corr_ratio_calc),
        'a_calc <= a_level': (a_corr_ratio_calc <= a_level),
        'significance_check': (conclusion_corr_ratio_sign),
        'conf_int_low': (corr_ratio_conf_int_low),
        'conf_int_high': (corr_ratio_conf_int_high),
        'scale': (conclusion_corr_ratio_scale)
        },
        index=['Correlation ratio'])
    
    return result      

    

#------------------------------------------------------------------------------
#   Функция line_corr_sign_check
#------------------------------------------------------------------------------

# Функция для проверка значимости линейной корреляционной связи

def line_corr_sign_check(X, Y, p_level=0.95, orientation='XY'):
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)    
    # коэффициент корреляции
    corr_coef = sci.stats.pearsonr(X, Y)[0]
    # корреляционное отношение
    try:
        if orientation!='XY' and orientation!='YX':
            raise ValueError("Error! Incorrect orientation!")
        if orientation=='XY':
            corr_ratio = corr_ratio_check(X, Y, orientation='XY', scale='Evans')['coef_value'].values[0]
        elif orientation=='YX':
            corr_ratio = corr_ratio_check(X, Y, orientation='YX', scale='Evans')['coef_value'].values[0]
    except ValueError as err:
        print(err)
    # число интервалов группировки
    group_int_number = lambda n: round (3.31*log(n_X, 10)+1) if round (3.31*log(n_X, 10)+1) >=2 else 2
    K_X = group_int_number(n_X)
    # проверка гипотезы о значимости линейной корреляционной связи
    if corr_ratio >= abs(corr_coef):
        F_line_corr_sign_calc = (n_X - K_X)/(K_X - 2) * (corr_ratio**2 - corr_coef**2) / (1 - corr_ratio**2)
        dfn = K_X - 2
        dfd = n_X - K_X
        F_line_corr_sign_table = sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
        comparison_F_calc_table = F_line_corr_sign_calc >= F_line_corr_sign_table
        a_line_corr_sign_calc = 1 - sci.stats.f.cdf(F_line_corr_sign_calc, dfn, dfd, loc=0, scale=1)
        comparison_a_calc_a_level = a_line_corr_sign_calc <= a_level
        conclusion_null_hypothesis_check = 'accepted' if F_line_corr_sign_calc < F_line_corr_sign_table else 'unaccepted'
        conclusion_line_corr_sign = 'linear' if conclusion_null_hypothesis_check == 'accepted' else 'non linear'
    else:
        F_line_corr_sign_calc = ''
        F_line_corr_sign_table = ''
        comparison_F_calc_table = ''
        a_line_corr_sign_calc = ''
        comparison_a_calc_a_level = ''
        conclusion_null_hypothesis_check = 'Attention! The correlation ratio is less than the correlation coefficient'
        conclusion_line_corr_sign = '-'
    
    # формируем результат            
    result = pd.DataFrame({
        'corr.coef.': (corr_coef),
        'corr.ratio.': (corr_ratio),
        'null hypothesis': ('r\u00b2 = ' + chr(951) + '\u00b2'),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_line_corr_sign_calc),
        'F_table': (F_line_corr_sign_table),
        'F_calc >= F_table': (comparison_F_calc_table),
        'a_calc': (a_line_corr_sign_calc),
        'a_calc <= a_level': (comparison_a_calc_a_level),
        'null_hypothesis_check': (conclusion_null_hypothesis_check),
        'significance_line_corr_check': (conclusion_line_corr_sign),
        },
        index=['Significance of linear correlation'])
    
    return result



#==============================================================================
#               ФУНКЦИИ ДЛЯ РЕГРЕССИОННОГО АНАЛИЗА
#==============================================================================


#------------------------------------------------------------------------------
#   Функция regression_error_metrics
#------------------------------------------------------------------------------

def regression_error_metrics(model, model_name=''):
    model_fit = model.fit()
    Ycalc = model_fit.predict()
    n_fit = model_fit.nobs
    Y = model.endog
    
    MSE = (1/n_fit) * np.sum((Y-Ycalc)**2)
    RMSE = sqrt(MSE)
    MAE = (1/n_fit) * np.sum(abs(Y-Ycalc))
    MSPE = (1/n_fit) *  np.sum(((Y-Ycalc)/Y)**2)
    MAPE = (1/n_fit) *  np.sum(abs((Y-Ycalc)/Y))
        
    model_error_metrics = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MSPE': MSPE,
        'MAPE': MAPE}
    
    result = pd.DataFrame({
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MSPE': "{:.3%}".format(MSPE),
        'MAPE': "{:.3%}".format(MAPE)},
        index=[model_name])        
        
    return model_error_metrics, result



#------------------------------------------------------------------------------
#   Функция ANOVA_table_regression_model
#------------------------------------------------------------------------------

def ANOVA_table_regression_model(model):
    n = int(model.nobs)
    p = int(model.df_model)
    SSR = model.ess
    SSE = model.ssr
    SST = model.centered_tss

    result = pd.DataFrame({
        'sources_of_variation': ('regression (SSR)', 'deviation from regression (SSE)', 'total (SST)'),
        'sum_of_squares': (SSR, SSE, SST),
        'degrees_of_freedom': (p, n-p-1, n-1)})
    result['squared_error'] = result['sum_of_squares'] / result['degrees_of_freedom']
    R2 = 1 - result.loc[1, 'sum_of_squares'] / result.loc[2, 'sum_of_squares']
    F_calc_adequacy = result.loc[2, 'squared_error'] / result.loc[1, 'squared_error']
    F_calc_determ_check = result.loc[0, 'squared_error'] / result.loc[1, 'squared_error']
    result['F-ratio'] = (F_calc_determ_check, F_calc_adequacy, '')
    
    return result



#------------------------------------------------------------------------------
#   Функция regression_model_adequacy_check
#------------------------------------------------------------------------------


def regression_model_adequacy_check(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    n = int(model_fit.nobs)
    p = int(model_fit.df_model)    # Число степеней свободы регрессии, равно числу переменных модели (за исключением константы, если она присутствует)
    
    SST = model_fit.centered_tss    # SST (Sum of Squared Total)
    dfT = n-1
    MST = SST / dfT

    SSE = model_fit.ssr    # SSE (Sum of Squared Error)
    dfE = n - p - 1
    MSE = SSE / dfE
    
    F_calc = MST / MSE
    F_table = sci.stats.f.ppf(p_level, dfT, dfE, loc=0, scale=1)
    a_calc = 1 - sci.stats.f.cdf(F_calc, dfT, dfE, loc=0, scale=1)
    conclusion_model_adequacy_check = 'adequacy' if F_calc >= F_table else 'adequacy'
    
    # формируем результат            
    result = pd.DataFrame({
        'SST': (SST),
        'SSE': (SSE),
        'dfT': (dfT),
        'dfE': (dfE),
        'MST': (MST),
        'MSE': (MSE),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_calc),
        'F_table': (F_table),
        'F_calc >= F_table': (F_calc >= F_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'adequacy_check': (conclusion_model_adequacy_check),
        },
        index=[model_name]
        )
    
    return result



#------------------------------------------------------------------------------
#   Функция determination_coef_check
#------------------------------------------------------------------------------

def determination_coef_check(
    model_fit,
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    R2 = model_fit.rsquared
    R2_adj = model_fit.rsquared_adj
    n = model_fit.nobs    # объем выборки
    p = model_fit.df_model    # Model degrees of freedom. The number of regressors p. Does not include the constant if one is present.
    
    F_calc = R2 / (1 - R2) * (n-p-1)/p
    df1 = int(p)
    df2 = int(n-p-1)
    F_table = sci.stats.f.ppf(p_level, df1, df2, loc=0, scale=1)
    a_calc = 1 - sci.stats.f.cdf(F_calc, df1, df2, loc=0, scale=1)
    conclusion_determ_coef_sign = 'significance' if F_calc >= F_table else 'not significance'
        
    # формируем результат            
    result = pd.DataFrame({
        'notation': ('R2'),
        'coef_value (R)': (sqrt(R2)),
        'coef_value_squared (R2)': (R2),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_calc),
        'df1': (df1),
        'df2': (df2),
        'F_table': (F_table),
        'F_calc >= F_table': (F_calc >= F_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'significance_check': (conclusion_determ_coef_sign),
        'conf_int_low': (''),
        'conf_int_high': ('')
        },
        index=['Coef. of determination'])
    return result



#------------------------------------------------------------------------------
#   Функция regression_coef_check
#------------------------------------------------------------------------------

def regression_coef_check(
    model_fit,
    notation_coef: list='',
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    # параметры модели (коэффициенты регрессии)
    model_params = model_fit.params
    # стандартные ошибки коэффициентов регрессии
    model_bse = model_fit.bse
    # проверка гипотезы о значимости регрессии
    t_calc = abs(model_params) / model_bse
    n = model_fit.nobs    # объем выборки
    p = model_fit.df_model    # Model degrees of freedom. The number of regressors p. Does not include the constant if one is present.
    df = int(n - p - 1)
    t_table = sci.stats.t.ppf((1 + p_level)/2 , df)
    a_calc = 2*(1-sci.stats.t.cdf(t_calc, df, loc=0, scale=1))
    conclusion_ = ['significance' if elem else 'not significance' for elem in (t_calc >= t_table).values]
        
    # доверительный интервал коэффициента регрессии
    conf_int_low = model_params - t_table*model_bse
    conf_int_high = model_params + t_table*model_bse
    
    # формируем результат            
    result = pd.DataFrame({
        'notation': (notation_coef),
        'coef_value': (model_params),
        'std_err': (model_bse),
        'p_level': (p_level),
        'a_level': (a_level),
        't_calc': (t_calc),
        'df': (df),
        't_table': (t_table),
        't_calc >= t_table': (t_calc >= t_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'significance_check': (conclusion_),
        'conf_int_low': (conf_int_low),
        'conf_int_high': (conf_int_high),
        })
    
    return result



#------------------------------------------------------------------------------
#   Функция Goldfeld_Quandt_test
#------------------------------------------------------------------------------

def Goldfeld_Quandt_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_goldfeldquandt(model_fit.resid, model_fit.model.exog)
    test_result = lzip(['F_statistic', 'p_calc'], test)    # распаковка результатов теста
    # расчетное значение статистики F-критерия
    F_calc_tuple = test_result[0]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости
    p_calc_tuple = test_result[1]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('Goldfeld–Quandt test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result



#------------------------------------------------------------------------------
#   Функция Breush_Pagan_test
#------------------------------------------------------------------------------

def Breush_Pagan_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_breuschpagan(model_fit.resid, model_fit.model.exog)
    name = ['Lagrange_multiplier_statistic', 'p_calc_LM', 'F_statistic', 'p_calc']
    test_result = lzip(name, test)    # распаковка результатов теста
    # расчетное значение статистики теста множителей Лагранжа
    LM_calc_tuple = test_result[0]
    Lagrange_multiplier_statistic = LM_calc_tuple[1]
    # расчетный уровень значимости статистики теста множителей Лагранжа
    p_calc_LM_tuple = test_result[1]
    p_calc_LM = p_calc_LM_tuple[1]
    # расчетное значение F-статистики гипотезы о том, что дисперсия ошибки не зависит от x
    F_calc_tuple = test_result[2]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости F-статистики
    p_calc_tuple = test_result[3]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'

    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('Breush-Pagan test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'Lagrange_multiplier_statistic': (Lagrange_multiplier_statistic),
        'p_calc_LM': (p_calc_LM),
        'p_calc_LM < a_level': (p_calc_LM < a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result



#------------------------------------------------------------------------------
#   Функция White_test
#------------------------------------------------------------------------------

def White_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_white(model_fit.resid, model_fit.model.exog)
    name = ['Lagrange_multiplier_statistic', 'p_calc_LM', 'F_statistic', 'p_calc']
    test_result = lzip(name, test)    # распаковка результатов теста
    # расчетное значение статистики теста множителей Лагранжа
    LM_calc_tuple = test_result[0]
    Lagrange_multiplier_statistic = LM_calc_tuple[1]
    # расчетный уровень значимости статистики теста множителей Лагранжа
    p_calc_LM_tuple = test_result[1]
    p_calc_LM = p_calc_LM_tuple[1]
    # расчетное значение F-статистики гипотезы о том, что дисперсия ошибки не зависит от x
    F_calc_tuple = test_result[2]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости F-статистики
    p_calc_tuple = test_result[3]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'

    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('White test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'Lagrange_multiplier_statistic': (Lagrange_multiplier_statistic),
        'p_calc_LM': (p_calc_LM),
        'p_calc_LM < a_level': (p_calc_LM < a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result


#------------------------------------------------------------------------------
#   Функция regression_pair_predict
#------------------------------------------------------------------------------


def regression_pair_predict(
    x_in,
    model_fit,
    regression_model,
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    X = pd.DataFrame(model_fit.model.exog)[1].values    # найти лучшее решение
    Y = model_fit.model.endog
    
    # вспомогательные величины
    n = int(result_linear_ols.nobs)
    SSE = model_fit.ssr    # SSE (Sum of Squared Error)
    dfE = n - p - 1
    MSE = SSE / dfE    # остаточная дисперсия
    
    Xmean = np.mean(X)
    SST_X = np.sum([(X[i] - Xmean)**2 for i in range(0, n)])
    
    t_table = sci.stats.t.ppf((1 + p_level)/2 , dfE)
    S2_y_calc_mean = MSE * (1/n + (x_in - Xmean)**2 / SST_X)
    S2_y_calc_predict = MSE * (1 + 1/n + (x_in - Xmean)**2 / SST_X)
        
    # прогнозируемое значение переменной Y
    y_calc=regression_model(x_in)
    # доверительный интервал средних значений переменной Y
    y_calc_mean_ci_low = y_calc - t_table*sqrt(S2_y_calc_mean)
    y_calc_mean_ci_upp = y_calc + t_table*sqrt(S2_y_calc_mean)
    # доверительный интервал индивидуальных значений переменной Y
    y_calc_predict_ci_low = y_calc - t_table*sqrt(S2_y_calc_predict)
    y_calc_predict_ci_upp = y_calc + t_table*sqrt(S2_y_calc_predict)
    
    result = y_calc, y_calc_mean_ci_low, y_calc_mean_ci_upp, y_calc_predict_ci_low, y_calc_predict_ci_upp
    
    return result


#------------------------------------------------------------------------------
#   Функция graph_regression_pair_predict_plot_sns
#------------------------------------------------------------------------------

def graph_regression_pair_predict_plot_sns(
    model_fit,
    regression_model_in,
    Xmin=None, Xmax=None, Nx=10,
    Ymin_graph=None, Ymax_graph=None,
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12, 
    label_legend_regr_model='', label_legend_fontsize=12,
    s=50, linewidth_regr_model=2,
    graph_size=(297/INCH, 210/INCH),
    result_output=True,
    file_name=None):
    
    # фактические данные
    X = pd.DataFrame(model_fit.model.exog)[1].values    # найти лучшее решение
    Y = model_fit.model.endog
    X = np.array(X)
    Y = np.array(Y)
    
    # границы
    if not(Xmin) and not(Xmax):
        Xmin=min(X)
        Xmax=max(X)
        Xmin_graph=min(X)*0.99
        Xmax_graph=max(X)*1.01
    else:
        Xmin_graph=Xmin
        Xmax_graph=Xmax
    
    if not(Ymin_graph) and not(Ymax_graph):
        Ymin_graph=min(Y)*0.99
        Ymax_graph=max(Y)*1.01       
    
    # формируем DataFrame данных
    Xcalc = np.linspace(Xmin, Xmax, num=Nx)
    Ycalc = regression_model_in(Xcalc)
    
    result_df = pd.DataFrame(
        [regression_pair_predict(elem, model_fit, regression_model=regression_model_in) for elem in Xcalc],
        columns=['y_calc', 'y_calc_mean_ci_low', 'y_calc_mean_ci_upp', 'y_calc_predict_ci_low', 'y_calc_predict_ci_upp'])
    result_df.insert(0, 'x_calc', Xcalc)
            
    # заголовки графика
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # фактические данные
    sns.scatterplot(
        x=X, y=Y,
        label='фактические данные',
        s=s,
        color='red',
        ax=axes)
    
    # график регрессионной модели
    sns.lineplot(
        x=Xcalc, y=Ycalc,
        color='blue',
        linewidth=linewidth_regr_model,
        legend=True,
        label=label_legend_regr_model,
        ax=axes)
    
    # доверительный интервал средних значений переменной Y
    Mean_ci_low = result_df['y_calc_mean_ci_low']
    plt.plot(
        result_df['x_calc'], Mean_ci_low,
        color='magenta', linestyle='--', linewidth=1,
        label='доверительный интервал средних значений Y')
    
    Mean_ci_upp = result_df['y_calc_mean_ci_upp']
    plt.plot(
        result_df['x_calc'], Mean_ci_upp,
        color='magenta', linestyle='--', linewidth=1)
    
    # доверительный интервал индивидуальных значений переменной Y
    Predict_ci_low = result_df['y_calc_predict_ci_low']
    plt.plot(
        result_df['x_calc'], Predict_ci_low,
        color='orange', linestyle='-.', linewidth=2,
        label='доверительный интервал индивидуальных значений Y')
    
    Predict_ci_upp = result_df['y_calc_predict_ci_upp']
    plt.plot(
        result_df['x_calc'], Predict_ci_upp,
        color='orange', linestyle='-.', linewidth=2)
    
        
    axes.set_xlim(Xmin_graph, Xmax_graph)
    axes.set_ylim(Ymin_graph, Ymax_graph)        
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    if result_output:
        return result_df
    else:
        return

    