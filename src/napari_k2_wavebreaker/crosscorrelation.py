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


def cycledegreesCross(input, pxpermicron, filename, restrictdeg, outputimg, outputpath):
    print("lets go")
    grid_a = input[1][0]
    grid_c = input[1][1]

    gridnumber = input[0]
    tempdict = {} # Dictionary to store the data of every degree
    df_grid = pd.DataFrame(columns=["deg", "periodicity_a", "frequency_a", "periodicity_c", "frequency_c", "crosscorlag", "gridindex", "pxpercentage"]) # Dataframe to store the data of every degree
    gridshape = np.shape(grid_a)
    gridpercentage = (np.size(grid_a) - np.count_nonzero(np.isnan(grid_a))) / np.size(grid_a)


    ### GRADIENT LABELS GENERATION ###
    angles = range(restrictdeg[0], restrictdeg[1]) # range of angles to be tested
    mask = np.ones(gridshape)
    mask[np.isnan(grid_a)] = 0

    labelslib = gengradient(angles, mask, gridshape, pxpermicron) # Generate gradients for every degree

    ### MEASURE LABELS FOR EVERY DEGREE ###

    for i in labelslib:
        data_a = measure.regionprops_table(labelslib[i]["labels"],
                                           grid_a,
                                           properties=['label', 'mean_intensity'])
        data_c = measure.regionprops_table(labelslib[i]["labels"],
                                           grid_c,
                                           properties=['label', 'mean_intensity'])

        ### CHECK FOR CONTINUEITY ###
        if checkConsecutive(data_a["label"]) == False or checkConsecutive(data_c["label"]) == False:
            pass
        else:
            ### CALCULATE CORRELATION ###
            tempdict[i] = {}  # Create a dictionary for every degree

            df_intensity = pd.DataFrame({"mean_intensity_a": data_a["mean_intensity"],
                                         "mean_intensity_c": data_c["mean_intensity"]})

            df_intensity["length"] = list(np.array(range(0, df_intensity.shape[0])) * labelslib[i]["pxoffset"]) # Generate length column normalized to the pxoffset based on the angle


            df_correlation = pd.DataFrame({"autocorr_a": autocorr(df_intensity["mean_intensity_a"], "Numpy"), # Calculate autocorrelation
                                           "autocorr_c": autocorr(df_intensity["mean_intensity_c"], "Numpy"), # Calculate autocorrelation
                                           "crosscorr": autocorr([df_intensity["mean_intensity_a"], df_intensity["mean_intensity_c"]], "Cross")}) # Calculate crosscorrelation

            lenindex = df_intensity.shape[0] - 1
            df_correlation["length"] = list(np.linspace(-lenindex, lenindex, 2 * lenindex + 1) * labelslib[i]["pxoffset"])  # Generate length column normalized to the pxoffset based on the angle

            ### EXTRACT FREQ, PERIODITY AND ADD TO ANGLE DICT ###
            cormin_a, cormax_a, periodicity_a, frequency_a = peakvalley(df_correlation["autocorr_a"], df_correlation["length"]) # Calculate periodicity and frequency of channel 1
            tempdict[i]["periodicity_a"] = periodicity_a
            tempdict[i]["frequency_a"] = frequency_a
            tempdict[i]["cormin_a"] = cormin_a
            tempdict[i]["cormax_a"] = cormax_a
            cormin_c, cormax_c, periodicity_c, frequency_c = peakvalley(df_correlation["autocorr_c"], df_correlation["length"]) # Calculate periodicity and frequency of channel 2
            tempdict[i]["periodicity_c"] = periodicity_c
            tempdict[i]["frequency_c"] = frequency_c
            tempdict[i]["cormin_c"] = cormin_c
            tempdict[i]["cormax_c"] = cormax_c
            crosscorlag = df_correlation["length"][middlepeak(df_correlation["crosscorr"])] # Calculate crosscorrelation lag
            tempdict[i]["crosscorlag"] = crosscorlag

            ### ADD RAW INTENSITY AND CORRELATION DATA TO TEMP DICT ###
            tempdict[i]["df_intensity"] = df_intensity
            tempdict[i]["df_correlation"] = df_correlation

            ### ADD DATA TO ANGLE DF ###
            df_angle = pd.DataFrame({"deg": [i],
                                      "periodicity_a": [periodicity_a],
                                      "frequency_a": [frequency_a],
                                      "periodicity_c": [periodicity_c],
                                      "frequency_c": [frequency_c],
                                      "crosscorlag": [crosscorlag],
                                      "gridindex": [filename + " / " + str(gridnumber)],
                                      "pxpercentage": [gridpercentage]})
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
            crosscorlag = tempdict[deg_at_best]["crosscorlag"]
            frequency_a = tempdict[deg_at_best]["frequency_a"]
            frequency_c = tempdict[deg_at_best]["frequency_c"]
            periodicity_a = tempdict[deg_at_best]["periodicity_a"]
            periodicity_c = tempdict[deg_at_best]["periodicity_c"]
            crosscorlag = tempdict[deg_at_best]["crosscorlag"]

            compactdf = pd.DataFrame(columns=["Deg", "Channel", "Periodicity", "Frequency", "Selected"])
            compactdf = pd.concat([compactdf, pd.DataFrame({"Deg": df_grid["deg"],
                                                            "Channel": "a",
                                                            "Periodicity": df_grid["periodicity_a"],
                                                            "Frequency": df_grid["frequency_a"]})])
            compactdf = pd.concat([compactdf, pd.DataFrame({"Deg": df_grid["deg"],
                                                            "Channel": "c",
                                                            "Periodicity": df_grid["periodicity_c"],
                                                            "Frequency": df_grid["frequency_c"]})])

            compactdf["Selected"] = np.where(compactdf["Deg"] == deg_at_best, True, False)

            compactdf = compactdf.dropna().reset_index(drop=True)

            fig, axs = plt.subplots(8, 1, figsize=(5, 14))
            sns.scatterplot(data=compactdf.where(compactdf["Channel"] == "a"), x="Deg", y="Frequency", hue="Periodicity", size="Periodicity", palette= sns.color_palette("dark:#3eb0db_r", as_cmap=True) , ax=axs[0], legend=False)
            sns.scatterplot(data=compactdf.where((compactdf["Channel"] == "a") & (compactdf["Selected"] == True)), x="Deg", y="Frequency", color="red", ax=axs[0], legend=False)
            axs[0].set_title("Frequency Disribution _ a")
            sns.scatterplot(data=compactdf.where(compactdf["Channel"] == "c"), x="Deg", y="Frequency", hue="Periodicity", size="Periodicity", palette= sns.color_palette("dark:#f35619_r", as_cmap=True) , ax=axs[1], legend=False)
            sns.scatterplot(data=compactdf.where((compactdf["Channel"] == "c") & (compactdf["Selected"] == True)), x="Deg", y="Frequency", color="red", ax=axs[1], legend=False)
            axs[1].set_title("Frequency Disribution _ c")
            axs[2].imshow(grid_a, cmap="gray")
            axs[2].set_title("Grid _ a")
            axs[3].imshow(grid_c, cmap="gray")
            axs[3].set_title("Grid _ c")
            axs[4].imshow(labelslib[deg_at_best]["labels"])
            axs[4].set_title(f"Labels gradient at {deg_at_best} degrees")

            sns.lineplot(data=df, x="length", y="mean_intensity_a", ax=axs[5], color="blue")
            sns.lineplot(data=df, x="length", y="mean_intensity_c", ax=axs[5], color="orange")

            axs[5].set_title("Intensity Profiles")
            axs[5].set_ylabel("Intensity")

            sns.lineplot(data=aut, x="length", y="autocorr_a", ax=axs[6], color="blue")
            sns.lineplot(data=aut, x="length", y="autocorr_c", ax=axs[6], color="orange")

            axs[6].axvline(x=frequency_a, color='blue', linestyle='-')
            axs[6].axvline(x=frequency_c, color='orange', linestyle='-')
            #add labels

            axs[6].set_title("P_a: " + str(round(periodicity_a,3)) + " / P_c: " + str(round(periodicity_c,3)))
            axs[6].set_ylabel("Correlation")

            sns.lineplot(data=aut, x="length", y="crosscorr", ax=axs[7], color="green")

            axs[7].axvline(x=crosscorlag, color='green', linestyle='-')

            axs[7].set_title("CC(a-c): " + str(round(crosscorlag,3)))

            plt.tight_layout()


            if not os.path.exists(outputpath + "/" + filename):
                os.mkdir(outputpath + "/" + filename)

            plt.savefig(outputpath + "/" + filename + "/" + str(gridnumber) + ".jpg")

    return df_grid

