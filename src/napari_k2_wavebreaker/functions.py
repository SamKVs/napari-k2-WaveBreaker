from typing import TYPE_CHECKING
import copy
import numpy as np
from math import isnan
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    elif method == "Cross":
        a = x[0]
        c = x[1]
        ccov = np.correlate(a - np.mean(a), c - np.mean(c), mode='full')
        ccor = ccov / (len(a) * np.std(a) * np.std(c))
        return ccor

def FFT(array,pixpermicron):
    FFT = np.fft.fft(array)
    x = np.array(range(len(FFT))) / pixpermicron

    plt.plot(x, FFT)




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