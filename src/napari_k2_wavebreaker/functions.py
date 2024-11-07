from typing import TYPE_CHECKING
import copy

import matplotlib.pyplot as plt
import numpy as np
from math import isnan
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.draw import line
from matplotlib.pyplot import xcorr
from skimage.feature import peak_local_max

if TYPE_CHECKING:
    import napari

def abspath(root, relpath):
    from pathlib import Path
    root = Path(root)
    if root.is_dir():
        path = root / relpath
    else:
        path = root.parent / relpath
    return str(path.absolute())

def checkConsecutive(l):
    return sorted(l) == list(range(min(l), max(l)+1))

def deg2vec(angle):
    return np.transpose(np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))

def slide(start, end, rails, mshape, mask):
    labels = np.zeros(mshape, dtype=int)
    color = 0
    if rails == 'y':
        while end[0] <= mshape[0]-1:
            rr, cc = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            coords = np.array(list(zip(rr, cc)))
            #clean up coordinates where either r or c is negative
            coords = coords[coords[:,0] <= mshape[0]-1]
            coords = coords[coords[:,1] <= mshape[1]-1]
            coords = coords[coords[:,0] >= 0]
            coords = coords[coords[:,1] >= 0]
            labels[coords[:,0], coords[:,1]] = color
            start[0] += 1
            end[0] += 1
            color += 1
    else:
        while end[1] <= mshape[1]-1:
            rr, cc = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            coords = np.array(list(zip(rr, cc)))
            #clean up coordinates where either r or c is negative
            coords = coords[coords[:,0] <= mshape[0]-1]
            coords = coords[coords[:,1] <= mshape[1]-1]
            coords = coords[coords[:,0] >= 0]
            coords = coords[coords[:,1] >= 0]
            labels[coords[:,0], coords[:,1]] = color
            start[1] += 1
            end[1] += 1
            color += 1

    labels[mask == 0] = np.ma.masked
    return labels

def gengradient(angles, mask, gridshape, pxpermicron):
    labelslib = {}
    vec = deg2vec(list(angles))
    rotvec = np.matmul(vec, np.array([[np.cos(np.radians(-45)), -np.sin(np.radians(-45))],
                                      [np.sin(np.radians(-45)), np.cos(np.radians(-45))]]))
    for index, i in enumerate(rotvec):
        tan = np.tan(np.radians(angles[index]))
        tan90 = np.tan(np.radians(angles[index] + 90))
        if all(i >= 0) or all(i < 0):
            rails = 'y'
            if tan >= 0:
                start = [0, gridshape[1] - 1]
                end = [-tan * gridshape[1], 0]
                pxoffset = abs(np.cos(np.radians(angles[index])) * (1/pxpermicron))
            else:
                start = [0, 0]
                end = [tan * gridshape[1], gridshape[1] - 1]
                pxoffset = abs(np.cos(np.radians(angles[index])) * (1/pxpermicron))
        else:
            rails = 'x'
            if tan >= 0:
                start = [gridshape[0] - 1, 0]
                end = [0, tan90 * gridshape[0]]
                pxoffset = abs(np.sin(np.radians(angles[index])) * (1/pxpermicron))
            else:
                start = [0, 0]
                end = [gridshape[0] - 1, - tan90 * gridshape[0]]
                pxoffset = abs(np.sin(np.radians(angles[index])) * (1/pxpermicron))

        labelslib[angles[index]] = {}
        labelslib[angles[index]]["labels"] = slide(np.around(start).astype(int), np.around(end).astype(int), rails, gridshape, mask)
        labelslib[angles[index]]["pxoffset"] = pxoffset

    return labelslib


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
                         color_continuous_scale=["#020024", "#024451", "#027371", "#028f92", "#03afb8", "#05d9b9",
                                                 "#05e69f"])
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
                            color_continuous_scale=["#020024", "#024451", "#027371", "#028f92", "#03afb8", "#05d9b9",
                                                    "#05e69f"])
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

    #if any of val is 0 set to the dimensiton of the array axis
    if val[0] == 0:
        val[0] = np.shape(array)[0]
    if val[1] == 0:
        val[1] = np.shape(array)[1]

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

        for row in range(len(rowsplit) - 1):
            for col in range(len(colsplit) - 1):
                grids.append(array[rowsplit[row]:rowsplit[row + 1], colsplit[col]:colsplit[col + 1]])

        return grids


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

        acorr = np.correlate(ndata, ndata, 'full')

        if var == 0:
            acorr = np.empty(len(acorr))[:]
            acorr[:] = np.nan
            return acorr

        acorr = acorr / var / len(ndata)


        return acorr

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

    elif method == "Cross":
        a = x[0]
        c = x[1]
        ccov = np.correlate(a - np.mean(a), c - np.mean(c), mode='full')

        if np.std(a) == 0 or np.std(c) == 0:
            ccor = np.empty(len(ccov))
            ccor[:] = np.nan
            return ccor

        ccor = ccov / (len(a) * np.std(a) * np.std(c))
        return ccor


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

def peakvalley(autocorlist, length):
    if autocorlist.isnull().values.all():
        return np.nan, np.nan, np.nan, np.nan
    autocorlist = np.array(autocorlist)
    length = np.array(length)
    # Determine all the peaks and values of the autocorrelation to calculate the frequency and periodicity
    cormin = (np.diff(np.sign(np.diff(autocorlist))) > 0).nonzero()[0] + 1  # local min
    cormax = (np.diff(np.sign(np.diff(autocorlist))) < 0).nonzero()[0] + 1  # local max
    if len(cormax) < 3 or len(cormin) < 2:
        periodicity = np.nan
        frequency = np.nan
        return cormin, cormax, periodicity, frequency
    else:
        maxpoint = autocorlist[cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1]]
        minpoint = autocorlist[cormin[int(len(cormin) / 2)]]

        periodicity = maxpoint - minpoint
        frequency = length[cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1]]

        return cormin, cormax, periodicity[0], frequency[0]

def middlepeak(crosscorlist):

    if crosscorlist.isnull().values.all():
        return np.nan

    def findlow(array):
        start = np.inf
        for x in array:
            if abs(x) <= abs(start):
                start = x
        return start

    crosscorlist = np.array(crosscorlist)
    cormax = (np.diff(np.sign(np.diff(crosscorlist))) < 0).nonzero()[0] + 1  # local max
    leng = len(crosscorlist)
    cormax = (cormax - (leng-1)/2)
    closest = []
    if len(cormax >= 1):
        closest.append(findlow(cormax))
        cormax = np.delete(cormax, np.where(cormax == closest[0]))
    if len(cormax >= 1):
        closest.append(findlow(cormax))
    closest = [x + (leng-1)/2 for x in closest]
    closest = list(map(int, closest))

    if len(closest) == 1:
        return closest[0]
    elif len(closest) == 0:
        return np.nan
    elif crosscorlist[closest[0]] >= crosscorlist[closest[1]]:
        return closest[0]
    else:
        return closest[1]

def localmaxcounter(array, pxpermicron):
    ## Get biological area from mask
    maskcount = np.count_nonzero(~np.isnan(array))
    maskarea = maskcount * ((1/ pxpermicron) ** 2)

    ## Get local maxima from raw grid data
    array = np.nan_to_num(array, nan=0)
    peakcount = len(peak_local_max(array, min_distance=1))

    ## Return points per area in micron
    return peakcount / maskarea


