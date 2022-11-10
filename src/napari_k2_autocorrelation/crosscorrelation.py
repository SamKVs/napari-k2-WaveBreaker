import os
import matplotlib.pyplot as plt
import numpy as np

from .functions import *

def peakvalley(autocorlist, pxpermicron_norm):
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
        frequency = (cormax[np.where(cormax == np.argmax(autocorlist))[0] + 1] -
                     len(autocorlist) / 2) / pxpermicron_norm
        return cormin, cormax, periodicity[0], frequency[0]

def middlepeak(crosscorlist):
    def findlow(array):
        start = np.inf
        for x in array:
            if abs(x) <= abs(start):
                start = x
        return start

    crosscorlist = np.array(crosscorlist)
    cormax = (np.diff(np.sign(np.diff(crosscorlist[0,:]))) < 0).nonzero()[0] + 1  # local max
    leng = len(crosscorlist[0,:])
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
    elif crosscorlist[0, closest[0]] >= crosscorlist[0, closest[1]]:
        return closest[0]
    else:
        return closest[1]


def cycledegreesCross(input, pxpermicron, filename, mode, restrictdeg, outputimg, outputcsv, outputpath):
    print("lets go")
    grid_a = input[1][0]
    grid_c = input[1][1]


    index = input[0]
    fitlist = []
    tempdict = {}
    dfPC = pd.DataFrame(columns=["deg", "periodicity_a", "frequency_a", "periodicity_c", "frequency_c", "crosscorlag", "gridindex", "pxpercentage"])
    gridshape = np.shape(grid_a)
    gridpercentage = (np.size(grid_a) - np.count_nonzero(np.isnan(grid_a))) / np.size(grid_a)

    # Determine the biological length and with of the image
    biorow = gridshape[0] / pxpermicron
    biocol = gridshape[1] / pxpermicron
    midangle = np.arctan(biorow / biocol)

    for deg in range(restrictdeg[0], restrictdeg[1]):

        tempdict[deg] = {}

        # From the defined angle calculate the coordinates where the midline goes through the border of the image
        intersect, intersectoposite = midline(grid_a, deg)

        originalintersect = copy.deepcopy(intersect)
        originalintersectoposite = copy.deepcopy(intersectoposite)

        x0 = intersect[1]
        x1 = intersectoposite[1]
        y0 = intersect[0]
        y1 = intersectoposite[0]

        meanarray_a = []
        meanarray_c = []

        # Starting at the midline intersect migrate the border points with a vector perpendicular to the defined degree.
        while checker(gridshape, intersect, intersectoposite):
            line = extractline(grid_a, intersect, intersectoposite)
            if np.isnan(line).all():
                meanarray_a.append(np.nan)
            else:
                meanarray_a.append(np.nanmean(line))

            line = extractline(grid_c, intersect, intersectoposite)
            if np.isnan(line).all():
                meanarray_c.append(np.nan)
            else:
                meanarray_c.append(np.nanmean(line))

            intersect, intersectoposite = nextpointsmart(gridshape, deg, intersect, intersectoposite, 'plus')

        # Same as above but now perpendicular to the defined degree in the other direction
        while checker(gridshape, originalintersect, originalintersectoposite):
            line = extractline(grid_a, originalintersect, originalintersectoposite)
            if np.isnan(line).all():
                meanarray_a.insert(0, np.nan)
            else:
                meanarray_a.insert(0, np.nanmean(line))

            line = extractline(grid_c, originalintersect, originalintersectoposite)
            if np.isnan(line).all():
                meanarray_c.insert(0, np.nan)
            else:
                meanarray_c.insert(0, np.nanmean(line))

            originalintersect, originalintersectoposite = nextpointsmart(gridshape, deg, originalintersect,
                                                                         originalintersectoposite, 'min')

        # if all points in the meanarray ar 0 or NaN, the array is ignored
        if not (np.isnan(np.nanmax(meanarray_a)) or np.nanmax(meanarray_a) == 0 or np.isnan(
                np.nanmax(meanarray_c)) or np.nanmax(meanarray_c) == 0):

            # Normalizes the user defined px/micron value based on the defined angle
            pxpermicron_norm = normalizepx(gridshape, midangle, deg, pxpermicron, len(meanarray_a))

            # Clean the NaN values at the edges of the meanarray
            newmeanarray_a = nanarraycleaner(meanarray_a)
            newmeanarray_c = nanarraycleaner(meanarray_c)

            # if there is still NaN values in the middle of the meanarray the array is discarded
            if not np.isnan(newmeanarray_a).any() and not np.isnan(newmeanarray_c).any():

                # Autocorrelation is done on the meanarray
                autocorlist_a = autocorr(newmeanarray_a, "Numpy")
                autocorlist_c = autocorr(newmeanarray_c, "Numpy")
                crosscorlist = autocorr([newmeanarray_a,newmeanarray_c], "Cross")
                crosscorarray = np.array([crosscorlist, (np.arange(-len(newmeanarray_a) + 1, len(newmeanarray_a))/pxpermicron_norm)])

                cormin_a, cormax_a, periodicity_a, frequency_a = peakvalley(autocorlist_a, pxpermicron_norm)
                fitlist.append([deg, periodicity_a, frequency_a])
                cormin_c, cormax_c, periodicity_c, frequency_c = peakvalley(autocorlist_c, pxpermicron_norm)
                crosscorlagindex = middlepeak(crosscorarray)
                crosscorlag = crosscorarray[1,crosscorlagindex]

                # Add all information to the tempdict
                tempdict[deg] = {
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "intensityplot_a": newmeanarray_a,
                    "intensityplot_c": newmeanarray_c,
                    "autocorrelationplot_a": autocorlist_a,
                    "autocorrelationplot_c": autocorlist_c,
                    "cormin_a": cormin_a,
                    "cormin_c": cormin_c,
                    "cormax_a": cormax_a,
                    "cormax_c": cormax_c,
                    "crosscorarray": crosscorarray,
                    "crosscorlagindex": crosscorlagindex,
                    "pxpermicron": pxpermicron_norm
                }

                if not (np.isnan(periodicity_a) or np.isnan(frequency_a) or np.isnan(periodicity_c) or np.isnan(frequency_c)):
                    tempdf = pd.DataFrame({"deg": [deg],
                                           "periodicity_a": [periodicity_a],
                                           "frequency_a": [frequency_a],
                                           "periodicity_c": [periodicity_c],
                                           "frequency_c": [frequency_c],
                                           "crosscorlag": [crosscorlag],
                                           "gridindex": [filename + " / " + str(index)],
                                           "pxpercentage": [gridpercentage]})
                    dfPC = pd.concat([dfPC, tempdf])

    fitlist = np.array(fitlist, dtype="float32")

    try:
        maxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 0]
        frequencyatmaxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 2]

    except:
        maxdeg = np.nan
        frequencyatmaxdeg = np.nan

        return dfPC, np.nan

    if outputimg:
        fig, axes = plt.subplots(nrows=6, figsize=(6, 11))
        axes[0].imshow(grid_a)
        axes[0].plot([tempdict[maxdeg]["x0"], tempdict[maxdeg]["x1"]], [tempdict[maxdeg]["y0"], tempdict[maxdeg]["y1"]],
                     'ro-')
        axes[0].axis('image')

        axes[1].imshow(grid_c)
        axes[1].plot([tempdict[maxdeg]["x0"], tempdict[maxdeg]["x1"]], [tempdict[maxdeg]["y0"], tempdict[maxdeg]["y1"]],
                     'ro-')
        axes[1].axis('image')


        autocorlist_a = tempdict[maxdeg]["autocorrelationplot_a"]
        autocorlist_c = tempdict[maxdeg]["autocorrelationplot_c"]
        micronlist_a = np.array(range(len(autocorlist_a))) / tempdict[maxdeg]["pxpermicron"]
        micronlist_c = np.array(range(len(autocorlist_c))) / tempdict[maxdeg]["pxpermicron"]
        micronlist_a = micronlist_a - (np.max(micronlist_a) / 2)
        micronlist_c = micronlist_c - (np.max(micronlist_c) / 2)
        micronlist2_a = np.array(range(len(tempdict[maxdeg]["intensityplot_a"]))) / tempdict[maxdeg]["pxpermicron"]
        micronlist2_c = np.array(range(len(tempdict[maxdeg]["intensityplot_c"]))) / tempdict[maxdeg]["pxpermicron"]


        axes[2].plot(micronlist2_a, tempdict[maxdeg]["intensityplot_a"])
        axes[2].plot(micronlist2_c, tempdict[maxdeg]["intensityplot_c"])

        axes[3].plot(micronlist_a, autocorlist_a)
        axes[4].plot(micronlist_c, autocorlist_c)

        axes[5].plot(tempdict[maxdeg]["crosscorarray"][1],tempdict[maxdeg]["crosscorarray"][0])
        axes[5].plot(tempdict[maxdeg]["crosscorarray"][1, tempdict[maxdeg]["crosscorlagindex"]],
                     tempdict[maxdeg]["crosscorarray"][0, tempdict[maxdeg]["crosscorlagindex"]],
                     marker="o", markersize=8, mec='r', mfc='r')

        plt.subplots_adjust(hspace=0.5)

        try:
            os.mkdir(outputpath + "/" + filename)
        except FileExistsError:
            pass

        plt.savefig(outputpath + "/" + filename + "/" + str(index) + ".jpg")

    return dfPC, np.nan
