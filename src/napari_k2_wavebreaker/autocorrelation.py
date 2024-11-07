import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure
import seaborn as sns

try:
    from .functions import *
except ImportError:
    from functions import *


def cycledegreesAuto(input, pxpermicron, filename, restrictdeg, outputimg, outputpath):
    print("lets go")
    grid = input[1]

    gridnumber = input[0]
    tempdict = {} # Dictionary to store the data of every degree
    df_grid = pd.DataFrame(columns=["deg", "periodicity_a", "frequency_a", "gridindex", "pxpercentage"]) # Dataframe to store the data of every degree
    gridshape = np.shape(grid)
    gridpercentage = (np.size(grid) - np.count_nonzero(np.isnan(grid))) / np.size(grid)
    gridpointcount = localmaxcounter(grid, pxpermicron)
    print(gridpointcount)


    ### GRADIENT LABELS GENERATION ###
    angles = range(restrictdeg[0], restrictdeg[1]) # range of angles to be tested
    mask = np.ones(gridshape)
    mask[np.isnan(grid)] = 0

    labelslib = gengradient(angles, mask, gridshape, pxpermicron) # Generate gradients for every degree

    ### MEASURE LABELS FOR EVERY DEGREE ###

    for i in labelslib:
        data_a = measure.regionprops_table(labelslib[i]["labels"],
                                           grid,
                                           properties=['label', 'mean_intensity'])

        ### CHECK FOR CONTINUEITY ###
        if checkConsecutive(data_a["label"]) == False:
            pass
        else:
            ### CALCULATE CORRELATION ###
            tempdict[i] = {}  # Create a dictionary for every degree

            df_intensity = pd.DataFrame({"mean_intensity_a": data_a["mean_intensity"]})

            df_intensity["length"] = list(np.array(range(0, df_intensity.shape[0])) * labelslib[i]["pxoffset"]) # Generate length column normalized to the pxoffset based on the angle


            df_correlation = pd.DataFrame({"autocorr_a": autocorr(df_intensity["mean_intensity_a"], "Numpy")}) # Calculate autocorrelation

            lenindex = df_intensity.shape[0] - 1
            df_correlation["length"] = list(np.linspace(-lenindex, lenindex, 2 * lenindex + 1) * labelslib[i]["pxoffset"])  # Generate length column normalized to the pxoffset based on the angle

            ### EXTRACT FREQ, PERIODITY AND ADD TO ANGLE DICT ###
            cormin_a, cormax_a, periodicity_a, frequency_a = peakvalley(df_correlation["autocorr_a"], df_correlation["length"]) # Calculate periodicity and frequency of channel 1
            tempdict[i]["periodicity_a"] = periodicity_a
            tempdict[i]["frequency_a"] = frequency_a
            tempdict[i]["cormin_a"] = cormin_a
            tempdict[i]["cormax_a"] = cormax_a

            ### ADD RAW INTENSITY AND CORRELATION DATA TO TEMP DICT ###
            tempdict[i]["df_intensity"] = df_intensity
            tempdict[i]["df_correlation"] = df_correlation

            ### ADD DATA TO ANGLE DF ###
            df_angle = pd.DataFrame({"deg": [i],
                                      "periodicity_a": [periodicity_a],
                                      "frequency_a": [frequency_a],
                                      "gridindex": [filename + " / " + str(gridnumber)],
                                      "pxpercentage": [gridpercentage],
                                      "pointspermicron2": [gridpointcount]})
            df_grid = pd.concat([df_grid, df_angle])


    ### PLOT THE DATA ###

    if outputimg:
        if df_grid["periodicity_a"].isnull().values.all():
            print("No data to plot")
        else:
            ### DETERMINE THE BEST ANGLE FOR THE PERIODICITY ###
            periodicity_a_best_index = np.nanargmax(list(df_grid["periodicity_a"]))
            deg_at_best = df_grid["deg"].iloc[periodicity_a_best_index]

            df = tempdict[deg_at_best]["df_intensity"]
            aut = tempdict[deg_at_best]["df_correlation"]
            frequency_a = tempdict[deg_at_best]["frequency_a"]
            periodicity_a = tempdict[deg_at_best]["periodicity_a"]

            compactdf = pd.DataFrame(columns=["Deg", "Channel", "Periodicity", "Frequency", "Selected"])
            compactdf = pd.concat([compactdf, pd.DataFrame({"Deg": df_grid["deg"],
                                                            "Channel": "a",
                                                            "Periodicity": df_grid["periodicity_a"],
                                                            "Frequency": df_grid["frequency_a"]})])

            compactdf["Selected"] = np.where(compactdf["Deg"] == deg_at_best, True, False)

            compactdf = compactdf.dropna().reset_index(drop=True)

            fig, axs = plt.subplots(5, 1, figsize=(5, 9))
            sns.scatterplot(data=compactdf.where(compactdf["Channel"] == "a"), x="Deg", y="Frequency", hue="Periodicity", size="Periodicity", palette= sns.color_palette("dark:#3eb0db_r", as_cmap=True) , ax=axs[0], legend=False)
            sns.scatterplot(data=compactdf.where((compactdf["Channel"] == "a") & (compactdf["Selected"] == True)), x="Deg", y="Frequency", color="red", ax=axs[0], legend=False)
            axs[0].set_title("Frequency Disribution _ a")
            axs[1].imshow(grid, cmap="gray")
            axs[1].set_title("Grid _ a")
            axs[2].imshow(labelslib[deg_at_best]["labels"])
            axs[2].set_title(f"Labels gradient at {deg_at_best} degrees")

            sns.lineplot(data=df, x="length", y="mean_intensity_a", ax=axs[3], color="blue")

            axs[3].set_title("Intensity Profiles")
            axs[3].set_ylabel("Intensity")

            sns.lineplot(data=aut, x="length", y="autocorr_a", ax=axs[4], color="blue")

            axs[4].axvline(x=frequency_a, color='blue', linestyle='-')

            #add labels

            axs[4].set_title("P_a: " + str(round(periodicity_a,3)))
            axs[4].set_ylabel("Correlation")

            plt.tight_layout()


            if not os.path.exists(outputpath + "/" + filename):
                os.mkdir(outputpath + "/" + filename)

            plt.savefig(outputpath + "/" + filename + "/" + str(gridnumber) + ".jpg")

    return df_grid

