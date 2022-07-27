"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QWidget, QFileDialog, QDockWidget
import qtpy.QtCore
from qtpy import uic
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from multiprocessing import Pool
from functools import partial
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
from napari.qt.threading import WorkerBase, WorkerBaseSignals
import math

if TYPE_CHECKING:
    import napari

from napari_plugin_engine import napari_hook_implementation
from pathlib import Path

def abspath(root, relpath):
    from pathlib import Path
    root = Path(root)
    if root.is_dir():
        path = root / relpath
    else:
        path = root.parent / relpath
    return str(path.absolute())


def PrincipleComponents(df, mode, highlight):
    features = ["deg", "periodicity", "frequency"]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    if mode == "2D PCA":
        pca = PCA(n_components=2)

        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf.reset_index(), df.reset_index()], axis=1)

        finalDf["periodicity"] = finalDf["periodicity"].astype(float)
        finalDf["frequency"] = finalDf["frequency"].astype(float)

        traceDf = finalDf[finalDf["frequency"].between(highlight[0], highlight[1])]

        fig = px.scatter(finalDf, x="principal component 1", y='principal component 2',
                         hover_data=["gridindex", "periodicity", "deg", "frequency"], color="periodicity",
                         color_continuous_scale=["#020024", "#024451", "#027371", "#028f92", "#03afb8", "#05d9b9", "#05e69f"])
        fig.add_traces(
            go.Scatter(x=traceDf["principal component 1"], y=traceDf['principal component 2'], mode="markers",
                       marker_symbol='star', marker_size=15, hoverinfo="none")
        )

    elif mode == "3D":

        df["periodicity"] = df["periodicity"].astype(float)
        df["frequency"] = df["frequency"].astype(float)

        traceDf = df[df["frequency"].between(highlight[0], highlight[1])]

        fig = px.scatter_3d(df, x="deg", y='frequency', z='periodicity',
                            hover_data=["gridindex", "periodicity", "deg", "frequency"], color="periodicity",
                            color_continuous_scale=["#020024", "#024451", "#027371", "#028f92", "#03afb8", "#05d9b9", "#05e69f"])
        fig.add_traces(
            go.Scatter3d(x=traceDf["deg"], y=traceDf['frequency'],
                         z=traceDf['periodicity'], mode="markers",
                         marker_symbol='diamond', marker_size=15, hoverinfo="skip")
        )

    fig.show()


def gridsplit(array, mode, val):
    def internalsplit(array, val):
        if val[0] == 0:
            val[0] = 1
        if val[1] == 0:
            val[1] = 1
        array = np.transpose(array)
        slices = np.array_split(array, val[1])
        grids = []
        for slice in slices:
            slice = np.transpose(slice)
            grido = np.array_split(slice, val[0])
            for grid in grido:
                grids.append(grid)
        return grids

    val = list(map(int, val))

    if mode == 'Manual':
        return internalsplit(array, val)

    if mode == 'Auto':
        arrayshape = np.shape(array)
        rowsplit = int(arrayshape[0] / val[0])
        colsplit = int(arrayshape[1] / val[1])
        return internalsplit(array, [rowsplit, colsplit])

    if mode == "None":
        return [array]

    if mode == "Fixed":
        arrayshape = np.shape(array)
        rowsplit = []
        colsplit = []
        grids = []

        rowdev = arrayshape[0] // val[0]
        coldev = arrayshape[1] // val[1]

        rowrest = arrayshape[0] % val[0]
        colrest = arrayshape[1] % val[1]

        for x in range(rowdev + 1):
            rowsplit.append(round((rowrest / 2) + (x * val[0])))

        for x in range(coldev + 1):
            colsplit.append(round((colrest / 2) + (x * val[1])))

        print(arrayshape)
        print(rowsplit)
        print(colsplit)

        for row in range(len(rowsplit) - 1):
            for col in range(len(colsplit) - 1):
                grids.append(array[rowsplit[row]:rowsplit[row + 1], colsplit[col]:colsplit[col + 1]])

        return grids

    if mode == "Custom":
        print(napari.layers.Shapes)


def autocorr(x, method):
    def misoformula(list, index):

        def DEVSQ(list):
            minmean = np.array(list) - np.average(list)
            return np.sum(minmean ** 2)

        count = len(list) / 3
        arraylen = len(list) - count
        static = np.array(list[0:int(arraylen - 1)]) - np.average(list)
        dynamic = np.array(list[index:int(arraylen + index - 1)]) - np.average(list)
        sumproduct = np.sum(static * dynamic)
        return sumproduct / DEVSQ(list)

    if method == "Numpy":
        x = np.array(x)

        # Mean
        mean = np.mean(x)

        # Variance
        var = np.var(x)

        # Normalized data
        ndata = x - mean

        acorr = np.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
        acorr = acorr / var / len(ndata)

        resultmin = np.flip(acorr[1:])
        result = np.concatenate((resultmin, acorr), axis=None)

        return result

    elif method == "Miso":
        x = (np.array(x) - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        resultplus = []
        for index, val in enumerate(x):
            count = len(x) / 3
            if index < count:
                resultplus.append(misoformula(x, index))

        resultmin = np.flip(resultplus[1:])
        result = np.concatenate((resultmin, resultplus), axis=None)
        return result


def midline(matrix, angle):
    midpoint = ((np.array(np.shape(matrix)) - 1) / 2).astype(int)
    r = np.tan(np.radians(angle))
    e = midpoint[0] - (r * midpoint[1])
    if r == 0:
        intersectx = np.inf
    else:
        intersectx = -e / r
    intersecty = e
    if 0 <= intersecty <= np.shape(matrix)[0] - 1:
        intersect = [int(intersecty), 0]
        intersectoposite = [(np.shape(matrix)[0] - 1) - int(intersecty), np.shape(matrix)[1] - 1]
    else:
        intersect = [0, int(intersectx)]
        intersectoposite = [np.shape(matrix)[0] - 1, (np.shape(matrix)[1] - 1) - int(intersectx)]

    return intersect, intersectoposite


def extractline(matrix, start, end):
    # -- Extract the line...
    # Make a line with "num" points...
    y0, x0 = start  # These are in _pixel_ coordinates!!
    y1, x1 = end
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # Extract the values along the line
    zi = matrix[y.astype('int'), x.astype('int')]
    return zi


def checker(matrixshape, start, end):
    if start[0] == 0 and start[0] == end[0]:
        return False
    elif start[0] == matrixshape[0] - 1 and start[0] == end[0]:
        return False
    elif start[1] == 0 and start[1] == end[1]:
        return False
    elif start[1] == matrixshape[1] - 1 and start[1] == end[1]:
        return False
    else:
        return True


def movevertical(point, vector):
    if vector[0] > 0:
        point[0] = point[0] + 1
    elif vector[0] < 0:
        point[0] = point[0] - 1

    return point


def movehorizontal(point, vector):
    if vector[1] > 0:
        point[1] = point[1] + 1
    elif vector[1] < 0:
        point[1] = point[1] - 1

    return point


def nextpointsmart(matrixshape, deg, start, end, direction):
    vectormin90 = [np.sin(np.radians(deg - 90)), np.cos(np.radians(deg - 90))]
    vectorplus90 = [np.sin(np.radians(deg + 90)), np.cos(np.radians(deg + 90))]

    limr = matrixshape[0] - 1
    limc = matrixshape[1] - 1

    if direction == 'plus':
        vector = vectorplus90
    else:
        vector = vectormin90

    appendo = []

    for point in [start, end]:
        realoriginalpoint = copy.deepcopy((point))
        originalpoint = copy.deepcopy(point)
        if point[0] in [0, limr]:
            proposedpoint = movehorizontal(point, vector)
            if proposedpoint[1] > limc or proposedpoint[1] < 0:
                newproposedpoint = movevertical(originalpoint, vector)
                if newproposedpoint[0] > limr or newproposedpoint[0] < 0:
                    appendo.append(realoriginalpoint)
                else:
                    appendo.append(newproposedpoint)
            else:
                appendo.append(proposedpoint)
        elif point[1] in [0, limc]:
            proposedpoint = movevertical(point, vector)
            if proposedpoint[0] > limr or proposedpoint[0] < 0:
                newproposedpoint = movehorizontal(originalpoint, vector)
                if newproposedpoint[1] > limc or newproposedpoint[1] < 0:
                    appendo.append(realoriginalpoint)
                else:
                    appendo.append(newproposedpoint)
            else:
                appendo.append(proposedpoint)

    return appendo[0], appendo[1]


def nanarraycleaner(list):
    list = np.array(list, dtype='float32')

    dellist = []
    i = 0
    j = len(list) - 1
    while isnan(list[i]):
        dellist.append(i)
        i = i + 1

    while isnan(list[j]):
        dellist.append(j)
        j = j - 1

    index_set = set(dellist)  # optional but faster
    output = [x for i, x in enumerate(list) if i not in index_set]

    return output


def normalizepx(arrayshape, midangle, deg, pxpermicron, pxamount):

    rad = np.radians(deg + 90)

    arrayshape = np.array(arrayshape) / pxpermicron

    lineangle = math.atan2(abs(math.sin(rad)), abs(math.cos(rad)))

    angle = abs(midangle - lineangle)

    hyp = np.hypot(arrayshape[1], arrayshape[0]) / 2

    length = (hyp * np.cos(angle)) * 2

    return pxamount / length


def cycledegrees(input, pxpermicron, filename, mode, restrictdeg, outputimg, outputcsv, outputpath):
    print("lets go")
    grid = input[1]
    index = input[0]
    fitlist = []
    tempdict = {}
    dfPC = pd.DataFrame(columns=["deg", "periodicity", "frequency", "gridindex"])
    gridshape = np.shape(grid)
    gridpercentage = (np.size(grid) - np.count_nonzero(np.isnan(grid))) / np.size(grid)

    biorow = gridshape[0] / pxpermicron
    biocol = gridshape[1] / pxpermicron

    midangle = np.arctan(biorow / biocol)

    for deg in range(restrictdeg[0], restrictdeg[1]):

        tempdict[deg] = {}

        intersect, intersectoposite = midline(grid, deg)

        originalintersect = copy.deepcopy(intersect)
        originalintersectoposite = copy.deepcopy(intersectoposite)

        x0 = intersect[1]
        x1 = intersectoposite[1]
        y0 = intersect[0]
        y1 = intersectoposite[0]

        meanarray = []

        while checker(gridshape, intersect, intersectoposite):
            if np.isnan(extractline(grid, intersect, intersectoposite)).all():
                meanarray.append(np.nan)
            else:
                meanarray.append(np.nanmean(extractline(grid, intersect, intersectoposite)))
            intersect, intersectoposite = nextpointsmart(gridshape, deg, intersect, intersectoposite, 'plus')

        while checker(gridshape, originalintersect, originalintersectoposite):
            if np.isnan(extractline(grid, originalintersect, originalintersectoposite)).all():
                meanarray.append(np.nan)
            else:
                meanarray.insert(0, np.nanmean(extractline(grid, originalintersect, originalintersectoposite)))
            originalintersect, originalintersectoposite = nextpointsmart(gridshape, deg, originalintersect,
                                                                         originalintersectoposite, 'min')

        if not (np.isnan(np.nanmax(meanarray)) or np.nanmax(meanarray) == 0):

            pxpermicron_norm = normalizepx(gridshape, midangle, deg, pxpermicron, len(meanarray))

            newmeanarray = nanarraycleaner(meanarray)

            if not np.isnan(newmeanarray).any():

                autocorlist = autocorr(newmeanarray, mode)


                cormin = (np.diff(np.sign(np.diff(autocorlist))) > 0).nonzero()[0] + 1  # local min
                cormax = (np.diff(np.sign(np.diff(autocorlist))) < 0).nonzero()[0] + 1  # local max
                if len(cormax) < 3 or len(cormin) < 2:
                    periodicity = np.nan
                    frequency = np.nan
                    fitlist.append([deg, periodicity, frequency])
                else:
                    maxpoint = autocorlist[cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1]]
                    minpoint = autocorlist[cormin[int(len(cormin) / 2)]]

                    periodicity = maxpoint - minpoint
                    frequency = (cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1] -
                                 len(autocorlist) / 2) / pxpermicron_norm
                    fitlist.append([deg, periodicity[0], frequency[0]])

                tempdict[deg] = {
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "intensityplot": newmeanarray,
                    "autocorrelationplot": autocorlist,
                    "cormin": cormin,
                    "cormax": cormax,
                    "pxpermicron": pxpermicron_norm
                }

                if not (np.isnan(periodicity) or np.isnan(frequency)):
                    tempdf = pd.DataFrame({"deg": [deg], "periodicity": [periodicity[0]], "frequency": [frequency[0]],
                                           "gridindex": [filename + " / " + str(index)], "pxpercentage": [gridpercentage]})
                    dfPC = pd.concat([dfPC, tempdf])

    fitlist = np.array(fitlist, dtype="float32")

    try:
        maxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 0]
        frequencyatmaxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 2]

    except ValueError:
        maxdeg = np.nan
        frequencyatmaxdeg = np.nan

        return np.nan, np.count_nonzero(np.isnan(grid)), frequencyatmaxdeg, dfPC

    if outputimg:
        fig, axes = plt.subplots(nrows=3)
        axes[0].imshow(grid)
        axes[0].plot([tempdict[maxdeg]["x0"], tempdict[maxdeg]["x1"]], [tempdict[maxdeg]["y0"], tempdict[maxdeg]["y1"]],
                     'ro-')
        axes[0].axis('image')

        cormin = tempdict[maxdeg]["cormin"]
        cormax = tempdict[maxdeg]["cormax"]
        autocorlist = tempdict[maxdeg]["autocorrelationplot"]
        micronlist = np.array(range(len(autocorlist))) / tempdict[maxdeg]["pxpermicron"]
        micronlist = micronlist - (np.max(micronlist)/2)
        micronlist2 = np.array(range(len(tempdict[maxdeg]["intensityplot"]))) / tempdict[maxdeg]["pxpermicron"]

        axes[1].plot(micronlist2, tempdict[maxdeg]["intensityplot"])
        axes[2].plot(micronlist, autocorlist)
        axes[2].plot((cormin / tempdict[maxdeg]["pxpermicron"]) - (np.max(micronlist)), autocorlist[cormin], "o", label="min", color='r')
        axes[2].plot((cormax / tempdict[maxdeg]["pxpermicron"]) - (np.max(micronlist)), autocorlist[cormax], "o", label="max", color='b')
        # axes[2].text(0.05, 0.95, np.nanmax(fitlist[:, 1]), transform=axes[2].transAxes, fontsize=14,
        #              verticalalignment='top')
        # axes[2].text(0.05, 0.7, frequencyatmaxdeg, transform=axes[2].transAxes, fontsize=14,
        #              verticalalignment='top')

        plt.subplots_adjust(hspace=0.5)

        try:
            os.mkdir(outputpath + "/" + filename)
        except FileExistsError:
            pass

        plt.savefig(outputpath + "/" + filename + "/" + str(index) + ".jpg")



    return np.nanmax(fitlist[:, 1]), np.size(grid) - np.count_nonzero(np.isnan(grid)), frequencyatmaxdeg, dfPC


# Define the main window classnapari
class AutocorrelationTool(QWidget):
    def __init__(self, napari_viewer):  # include napari_viewer as argument (it has to have this name)
        super().__init__()

        self.viewer = napari_viewer
        self.UI_FILE = abspath(__file__, 'static/form.ui')  # path to .ui file
        uic.loadUi(self.UI_FILE, self)

        self.viewer.layers.events.removed.connect(self.updatelayer)
        self.viewer.layers.events.inserted.connect(self.updatelayer)
        self.viewer.layers.events.changed.connect(self.updatelayer)

        self.thread = None
        self.inputarray = None
        self.maskedarray = None
        self.outputPath = None
        self.restrictdeg = [-90, 90]

        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))

        self.genThresh.clicked.connect(self.visualizeThresh)
        self.pushButton_genShapes.clicked.connect(self.createshapelayer)

        self.comboBox_mode.currentIndexChanged.connect(self.changeLock_zone)
        self.comboBox_gridsplit.currentIndexChanged.connect(self.changeLock_grid)
        self.comboBox_visOutput.currentIndexChanged.connect(self.changeLock_vis)
        self.angleSlider.valueChanged.connect(self.updateslidervalue)
        self.analyze.clicked.connect(self.Autocorrelate)
        self.pushButton_File.clicked.connect(self.filedialog)


    def updateslidervalue(self):
        self.sliderLabel.setText(str(self.angleSlider.value()))

    def filedialog(self):
        self.outputPath = QFileDialog.getExistingDirectory(self, 'Select output path')

    def changeLock_zone(self):
        if self.comboBox_mode.currentText() == "Full search":
            self.spinBox_zoneMid.setEnabled(False)
            self.angleSlider.setEnabled(False)
            self.restrictdeg = [-90, 90]

        else:
            self.spinBox_zoneMid.setEnabled(True)
            self.angleSlider.setEnabled(True)
            self.restrictdeg = [self.spinBox_zoneMid.value() - self.angleSlider.value(),
                                self.spinBox_zoneMid.value() + self.angleSlider.value()]

    def changeLock_grid(self):
        if self.comboBox_gridsplit.currentText() == "None":
            self.spinBox_gridLeft.setEnabled(False)
            self.spinBox_gridRight.setEnabled(False)

        else:
            self.spinBox_gridLeft.setEnabled(True)
            self.spinBox_gridRight.setEnabled(True)

    def changeLock_vis(self):
        if self.comboBox_visOutput.currentText() == "None":
            self.doubleSpinBox_visLeft.setEnabled(False)
            self.doubleSpinBox_visRight.setEnabled(False)

        else:
            self.doubleSpinBox_visLeft.setEnabled(True)
            self.doubleSpinBox_visRight.setEnabled(True)

    def updatelayer(self):
        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))

    def threshold(self):
        ### Set type, blur and threshold
        self.maskedarray = copy.deepcopy(self.inputarray)
        blurredz = np.array(self.inputarray, dtype='uint8')
        blurredz = cv2.GaussianBlur(blurredz, (5, 5), 5)

        ### MANUAL THRESHOLDING ###
        thresh = self.threshSlider.value()
        thresholdH = blurredz[:, :] > thresh
        thresholdL = blurredz[:, :] <= thresh
        blurredz[thresholdH] = 1
        blurredz[thresholdL] = 0

        edged = cv2.Canny(blurredz, 0, 1)

        contours, hierarchy = cv2.findContours(blurredz, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(blurredz.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        ### overlay original image with mask
        self.maskedarray[mask == 0] = np.nan

        return mask

    def readfile(self):
        ### Read selected layer
        inputarray = self.viewer.layers[self.comboBox_layer.currentText()].data
        ### Set to greyscale if needed
        try:
            inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass

        print(inputarray)

        self.inputarray = np.array(inputarray, dtype='float32')

    def visualizeThresh(self):
        self.readfile()
        mask = self.threshold()
        self.viewer.add_image(mask,
                              name=str(self.viewer.layers[self.comboBox_layer.currentText()]) + "_mask",
                              colormap="cyan",
                              opacity=0.30)

    def createshapelayer(self):
        self.viewer.add_shapes(shape_type="rectangle", edge_width=5, edge_color='#05d9b9', face_color='#05e69f',
                               opacity=0.4, name=str(self.viewer.layers[self.comboBox_layer.currentText()]) + "_ROI")

    def updateprogress(self, progress):
        if progress[0]:
            self.progressBar.setValue(100)
        else:
            self.progressBar.setValue(self.progressBar.value() + (1 / int(progress[1])) * 100)

    def Autocorrelate(self):
        # print(self.viewer.layers[self.comboBox_layer.currentText() + "_ROI"].data)
        self.readfile()
        self.threshold()
        self.changeLock_zone()
        self.thread = MyWorker()
        self.thread.updateparameters(currentlayer=self.comboBox_layer.currentText(),
                                     maskedarray=self.maskedarray,
                                     mode=self.comboBox_mode.currentText(),
                                     gridsplitmode=self.comboBox_gridsplit.currentText(),
                                     gridsplitleft=self.spinBox_gridLeft.value(),
                                     gridsplitright=self.spinBox_gridRight.value(),
                                     autocormode=self.comboBox_AutocorMethod.currentText(),
                                     visoutput=self.comboBox_visOutput.currentText(),
                                     visleft=self.doubleSpinBox_visLeft.value(),
                                     visright=self.doubleSpinBox_visRight.value(),
                                     pixelsize=self.spinBox_pixel.value(),
                                     outimg=self.checkBox_outImg.isChecked(),
                                     outcsv=self.checkBox_outCSV.isChecked(),
                                     path=self.outputPath,
                                     restrictdeg=self.restrictdeg)

        self.progressBar.setValue(0)
        self.thread.start()

        self.thread.progress.connect(self.updateprogress)



class MyWorkerSignals(WorkerBaseSignals):
    progress = qtpy.QtCore.Signal(object)

class MyWorker(WorkerBase):
    # progress = qtpy.QtCore.Signal(object)

    def __init__(self):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.currentlayer = None
        self.path = None
        self.outcsv = None
        self.outimg = None
        self.pixelsize = None
        self.visright = None
        self.visleft = None
        self.visoutput = None
        self.autocormode = None
        self.gridsplitright = None
        self.gridsplitleft = None
        self.gridsplitmode = None
        self.analysismode = None
        self.maskedarray = None
        self.restrictdeg = None

    def updateparameters(self,
                         currentlayer,
                         maskedarray,
                         mode,
                         gridsplitmode,
                         gridsplitleft,
                         gridsplitright,
                         autocormode,
                         visoutput,
                         visleft,
                         visright,
                         pixelsize,
                         outimg,
                         outcsv,
                         path,
                         restrictdeg):

        self.currentlayer = currentlayer
        self.maskedarray = maskedarray
        self.analysismode = mode
        self.gridsplitmode = gridsplitmode
        self.gridsplitleft = gridsplitleft
        self.gridsplitright = gridsplitright
        self.autocormode = autocormode
        self.visoutput = visoutput
        self.visleft = visleft
        self.visright = visright
        self.pixelsize = pixelsize
        self.outimg = outimg
        self.outcsv = outcsv
        self.path = path
        self.restrictdeg = restrictdeg

    def work(self):
        print("ok")
        gridsplitmode = self.gridsplitmode
        # gridsplitmode = "Auto"

        gridsplitval = [self.gridsplitleft, self.gridsplitright]
        # gridsplitval = [200,200]

        cleangrids = []

        grids = gridsplit(self.maskedarray, gridsplitmode, gridsplitval)
        for index, grid in enumerate(grids):
            if not np.isnan(grid).all():
                cleangrids.append(grid)

        indexgrids = []

        for index, grid in enumerate(cleangrids):
            indexgrids.append([index, grid])

        with Pool(4) as self.pool:
            output = []
            for _ in self.pool.imap_unordered(partial(cycledegrees,
                                                      pxpermicron=self.pixelsize,
                                                      filename=self.currentlayer,
                                                      mode=self.autocormode,
                                                      outputimg=self.outimg,
                                                      outputcsv=self.outcsv,
                                                      restrictdeg=self.restrictdeg,
                                                      outputpath=self.path), indexgrids):
                output.append(_)
                self.progress.emit([False, len(indexgrids)])
                # print(self.progressBar.value())
                # self.progressBar.setValue(self.progressBar.value() + 1)

            self.pool.close()
            self.pool.join()

            self.progress.emit([True, len(indexgrids)])
            output = np.array(output)
            weighted_avg = np.average(output[:, 0], weights=output[:, 1])
            intervallist = output[:, 2]
            medianfrequency = np.average(output[:, 2], weights=output[:, 1])
            print('FINAL RESULT', weighted_avg)
            print(intervallist)
            print('most likely periodicity interval', medianfrequency)

            if not self.visoutput == "None":
                df = pd.concat(output[:, 3])
                PrincipleComponents(df, self.visoutput,
                                    (self.visleft, self.visright))
            if not self.outcsv == "None":
                df = pd.concat(output[:, 3])
                df2 = pd.DataFrame({"total grids": [np.shape(indexgrids)[0]]})
                new = pd.concat([df, df2], axis=1)
                new.to_csv(self.path + "/" + self.currentlayer + ".csv", sep= ";")

    def stop(self):
        self.terminate()
        self.pool.stop()



@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return AutocorrelationTool
