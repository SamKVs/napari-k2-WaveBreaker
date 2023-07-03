
import os
import matplotlib.pyplot as plt
try:
    from .functions import *
except ImportError:
    from functions import *

def cycledegreesAuto(input, pxpermicron, filename, mode, restrictdeg, outputimg, outputcsv, outputpath):
    print("lets go")
    grid = input[1]
    index = input[0]
    fitlist = []
    tempdict = {}
    dfPC = pd.DataFrame(columns=["deg", "periodicity", "frequency", "gridindex"])
    gridshape = np.shape(grid)
    gridpercentage = (np.size(grid) - np.count_nonzero(np.isnan(grid))) / np.size(grid)

    # Determine the biological length and with of the image
    biorow = gridshape[0] / pxpermicron
    biocol = gridshape[1] / pxpermicron
    midangle = np.arctan(biorow / biocol)

    for deg in range(restrictdeg[0], restrictdeg[1]):

        tempdict[deg] = {}

        # From the defined angle calculate the coordinates where the midline goes through the border of the image
        intersect, intersectoposite = midline(grid, deg)

        originalintersect = copy.deepcopy(intersect)
        originalintersectoposite = copy.deepcopy(intersectoposite)

        x0 = intersect[1]
        x1 = intersectoposite[1]
        y0 = intersect[0]
        y1 = intersectoposite[0]

        meanarray = []

        # Starting at the midline intersect migrate the border points with a vector perpendicular to the defined degree.
        while checker(gridshape, intersect, intersectoposite):
            if np.isnan(extractline(grid, intersect, intersectoposite)).all():
                meanarray.append(np.nan)
            else:
                meanarray.append(np.nanmean(extractline(grid, intersect, intersectoposite)))
            intersect, intersectoposite = nextpointsmart(gridshape, deg, intersect, intersectoposite, 'plus')

        # Same as above but now perpendicular to the defined degree in the other direction
        while checker(gridshape, originalintersect, originalintersectoposite):
            if np.isnan(extractline(grid, originalintersect, originalintersectoposite)).all():
                meanarray.append(np.nan)
            else:
                meanarray.insert(0, np.nanmean(extractline(grid, originalintersect, originalintersectoposite)))
            originalintersect, originalintersectoposite = nextpointsmart(gridshape, deg, originalintersect,
                                                                         originalintersectoposite, 'min')

        # if all points in the meanarray ar 0 or NaN, the array is ignored
        if not (np.isnan(np.nanmax(meanarray)) or np.nanmax(meanarray) == 0):

            # Normalizes the user defined px/micron value based on the defined angle
            pxpermicron_norm = normalizepx(gridshape, midangle, deg, pxpermicron, len(meanarray))

            # Clean the NaN values at the edges of the meanarray
            newmeanarray = nanarraycleaner(meanarray)

            # if there is still NaN values in the middle of the meanarray the array is discarded
            if not np.isnan(newmeanarray).any():

                # Autocorrelation is done on the meanarray
                autocorlist = autocorr(newmeanarray, mode)

                # Determine all the peaks and values of the autocorrelation to calculate the frequency and periodicity
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

                # Add all information to the tempdict
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
                                           "gridindex": [filename + " / " + str(index)],
                                           "pxpercentage": [gridpercentage]})
                    dfPC = pd.concat([dfPC, tempdf])

    fitlist = np.array(fitlist, dtype="float32")

    try:
        maxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 0]
        frequencyatmaxdeg = fitlist[np.nanargmax(fitlist[:, 1]), 2]
    except Exception:
        maxdeg = np.nan
        frequencyatmaxdeg = np.nan

        return dfPC

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
        micronlist = micronlist - (np.max(micronlist) / 2)
        micronlist2 = np.array(range(len(tempdict[maxdeg]["intensityplot"]))) / tempdict[maxdeg]["pxpermicron"]

        axes[1].plot(micronlist2, tempdict[maxdeg]["intensityplot"])
        axes[2].plot(micronlist, autocorlist)

        plt.subplots_adjust(hspace=0.5)

        try:
            os.mkdir(outputpath + "/" + filename)
        except FileExistsError:
            pass

        plt.savefig(outputpath + "/" + filename + "/" + str(index) + ".jpg")

    return dfPC
