import pandas as pd
import os
from pandas import DataFrame
import sys
import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt


# LMS 계측 정보 추출하는 함수
'''
def m_info(rawdata):
    a = rawdata.index[rawdata[0].isna()]  # NA행 찾기 (1행, 112행)
    li = rawdata.index[rawdata[0] == 'Linear']  # 계측 데이터 찾기
    mea_info = rawdata[a[0] + 1:a[1]]
    mea_spectrum = rawdata.iloc[li[0]+1:]
    mea_data = pd.concat(mea_info, mea_spectrum)
    #m2 = mea_info.drop(mea_info.columns[2:], axis='columns')  # measurement information
    split_words = mea_info[0].str.split("\\").str[-1]  # 완전 힘들게 찾았음 ㅎㅎㅎ
    mea_info[0] = split_words
    mea_info_modi = mea_info.drop_duplicates([0]).sort_values(by=[0], axis=0)  # 0번 열 오름차순 정리
    select_info = ['Actual sensitivity', 'Average type', 'Channelgroup', 'DOF id', 'Frequency resolution',
                   'Function class', 'HW Range', 'Measured quantity', 'Number of averages', 'Number of lines',
                   'Original project', 'Original section', 'Overlap', 'Spectrum scaling', 'Window type',
                   'Y axis unit', 'Original run']              # 필요 계측정보 - 수정가능
    mea_info_last = mea_info_modi[mea_info_modi[0].isin(select_info)].reset_index(drop=True)    # 최종 계측 정보
    return mea_info_last

#def m_data(rawdata):
'''

rawdata = pd.read_csv('u24a.csv', header=None, encoding='cp949')

naraw = rawdata.index[rawdata[0].isna()]  # NA행 찾기 (1행, 112행)
li = rawdata.index[rawdata[0] == 'Linear']  # 계측 데이터 찾기
mea_info = rawdata[naraw[0] + 1:naraw[1]].T.drop_duplicates().T
split_words = mea_info[0].str.split("\\").str[-1]  # 완전 힘들게 찾았음 ㅎㅎㅎ
mea_info[0] = split_words
mea_info_modi = mea_info.drop_duplicates([0]).sort_values(by=[0], axis=0)  # 0번 열 오름차순 정리
select_info = ['Actual sensitivity', 'Average type', 'Channelgroup', 'DOF id', 'Frequency resolution',
               'Function class', 'HW Range', 'Measured quantity', 'Number of averages', 'Number of lines',
               'Original project', 'Original section', 'Overlap', 'Spectrum scaling', 'Window type',
               'Y axis unit', 'Original run']  # 필요 계측정보 - 수정가능

mea_info_last = mea_info_modi[mea_info_modi[0].isin(select_info)]  # 최종 계측 정보

mea_spectrum = rawdata.iloc[li[0] + 1:].T.drop_duplicates().T
mea_data = pd.concat([mea_info_last, mea_spectrum], ignore_index=True).reset_index(drop=True)
mea_data.to_csv('u24a_saved.csv',header=False, index=False, encoding='cp949')
b = mea_data.index[mea_data[0] == 'Original run']    # original run 행 인덱스 추출
mea_spectrum.columns = mea_data.iloc[b[0],:]
mea_spectrum = mea_spectrum.set_index('Original run')
mea_spectrum = mea_spectrum.astype(float)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)

ax.plot(mea_spectrum.index.astype(float), mea_spectrum)
plt.show()