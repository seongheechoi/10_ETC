import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):

        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.setWindowIcon(QIcon('icon.png'))

        self.lineEdit = QLineEdit()
        self.pushButton = QPushButton("Graph 그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.lineEdit)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def pushButtonClicked(self):

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
        mea_data.to_csv('u24a_saved.csv', header=False, index=False, encoding='cp949')
        b = mea_data.index[mea_data[0] == 'Original run']  # original run 행 인덱스 추출
        mea_spectrum.columns = mea_data.iloc[b[0], :]
        mea_spectrum = mea_spectrum.set_index('Original run')
        mea_spectrum = mea_spectrum.astype(float)

        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(mea_spectrum.index.astype(float), mea_spectrum)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()