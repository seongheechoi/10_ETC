import sys

#from pyqtgraph.Qt import QtGui, QtCore     # ---
from PyQt5 import QtWidgets, QtGui, QtCore  # +++

import numpy as np
import pyqtgraph as pg
import time
import random

class TestClass(QtGui.QMainWindow):
  #####################################################
  def __init__(self):
    super(TestClass, self).__init__()
    self.num = 0                        ### +++
    self.initUI()

  #####################################################
  # GUI construction
  def initUI(self):
    win = pg.GraphicsWindow(title="Mouse Point, x & y")

    # creates plot
    self.plot = pg.PlotWidget()
    self.plot.setLabel('left', "B", units='T')
    self.plot.setLabel('bottom', "t", units='s')
    self.plot.showGrid(x=1, y=1, alpha=None)
    self.setCentralWidget(win)
    self.setGeometry(600, 600, 600, 600)
    self.setWindowTitle('Mouse Point, x& y GUI')

    # Create some widgets to be placed inside
    btnRandon = QtGui.QPushButton('Random Function')


    # Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    win.setLayout(layout)

    # Add widgets to the layout in their proper positions
    layout.addWidget(btnRandon, 0, 0) # button to show or hide the OldB
    layout.addWidget(self.plot, 1, 0)

    mypen = pg.mkPen('y', width=1)
    self.curve = self.plot.plot(pen=mypen)

    # Plot
    self.t_plot_max = 30
    self.fe = 10e3
    self.t = np.arange(-1 * self.t_plot_max, 0, 1.0 / self.fe)
    self.len_signal = len(self.t)
    self.signal = np.zeros(self.len_signal, dtype=np.double)

    # status bar
    self.statusBar()

    # clicked button evt
    btnRandon.clicked.connect(self.buttonRandomClicked)

    # show graph
    self.show()

#  ### ------------------------------------------------
#  def mouseMoved(evt):
#    mousePoint = self.curve.vb.mapSceneToView(evt[0])
#    label.setText("<span style='font-size: 14pt; color: white'> x = %0.2f, <span style='color: white'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))


  #####################################################
  def buttonRandomClicked(self):
    print ("Show/Hide OldB")
    self.signal = np.random.rand(20)
    self.curve.setData(self.signal)

#####################################################
#  def update(self):
#    proxy = pg.SignalProxy(self.curve.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
#    self.statusBar().showMessage('Update timer event')
### vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  def update(self):
    ### +++++++++++++++++++++++++++++++++++++++++++++
    def mouseMoved(evt):
      print("-->> def mouseMoved(evt):", evt)
      print("\tevt.x()=`{}`, evt.y()=`{}`".format(evt.x(), evt.y()))

      # AttributeError: 'PlotDataItem' object has no attribute 'vb'        ### ???????
      #mousePoint = self.curve.vb.mapSceneToView(evt[0])

      # vvvv - > label what is it? <-- NameError: name `label` is not defined ### ???????
      #label.setText("<span style='font-size: 14pt; color: white'> x = %0.2f, <span style='color: white'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))


    self.source = self.curve.scene().sigMouseMoved
    #print(" source ", self.source)
    proxy = pg.SignalProxy(self.source, rateLimit=60, slot=mouseMoved) #+self
    #print("def update(self):222", proxy)
    if self.source is None:
        pass
        #sp.connect(sp, QtCore.SIGNAL('signal'), slot)
    else:
        #sp.connect(sp, signal, slot)
        proxy.signal.connect(mouseMoved)

    self.statusBar().showMessage('Update timer event `{}`'.format(self.num))
    self.num += 1
    return proxy



# MAIN ##################################################
def main():
  app = QtGui.QApplication(sys.argv)
  ex = TestClass()
  timer = QtCore.QTimer()
  timer.timeout.connect(ex.update)
  timer.start(200)
  sys.exit(app.exec_())


if __name__ == '__main__':
  main()