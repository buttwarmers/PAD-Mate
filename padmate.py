# -*- coding: utf-8 -*-

# =============================================================================
# 
# PAD Mate is a helper app for the mobile game Puzzle & Dragons.
# 
# =============================================================================

import sys
import os
import ctypes

from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSlot, pyqtSignal

import pyqt5ac

from modules import build, utils, texturetool

# Find and load the resource file
pyqt5ac.main(config='resources/resources config.yml')

# Give Windows 10 information so it can set the taskbar icon correctly
app_name = 'PAD Mate'
app_id = '{app_name}.{app_name}.{app_name}.{app_name}'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

# Load the UI file
form_class = uic.loadUiType('gui/FRHEED.ui')[0]

# Initialize the QApplication
app = QApplication(sys.argv)

class PadMate(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

class WorkerSignals(QObject):
    result = pyqtSignal(object)
  
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
        self.kwargs['result'] = self.signals.result
        
    @pyqtSlot()
    def run(self):
        self.running = True
        self.fn(*self.args, **self.kwargs)

if __name__ == '__main__':
    PadMate()
    app.exec_()

