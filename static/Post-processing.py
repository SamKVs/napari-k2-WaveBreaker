import numpy as np
import os
import pandas as pd


#SETUP

# Python file should be copied and run in the same folder as the csv files gathered from the Wavebreaker plugin
root = os.path.dirname(os.path.abspath(__file__))
# Set the frequency range (in µm) to be considered for the analysis
freqrange = [0.17, 0.23]
# Set the angle range (in degrees) to be considered for the analysis
#   (if an appropriate angle range was already set during analysis, keep degrange to [-180, 180])
degrange = [-180, 180]
# If doing crosscorrelation between two channels, perlimit is the minimum autocorrelation
#   amplitude for both channels to be considered for crosscorrelation
#   (if perlimit is set to 0, all channels will be considered)
#   (This parameter is ignored when csv files are from autocorelation analysis only)
perlimit = 0
# Set the output file name for the post-processing analysis (without extension)
#   (File will always be *.csv and saved in the same folder as the input csv files and this python file)
#   (out_name cannot contain / or \)
out_name = 'summary'


def CCnorm(x, freq):
    half = freq/2
    x = abs(x%freq)
    x = x - half
    x = abs(x)
    x = -x + half
    return x




final = pd.DataFrame()
finalCC = pd.DataFrame()

for file in os.listdir(root):
    if file.endswith(".csv"):
        data = pd.read_csv(os.path.join(root,file), sep=';')
        experiment = data['gridindex'].iloc[0].split('/')[0]
        data['grid'] = data['gridindex'].apply(lambda x: int(x.split('/')[1]))
        gridlist = data['grid'].unique()
        print(data.columns)

        data = data[data['deg'] > degrange[0]]
        data = data[data['deg'] < degrange[1]]

        for channel in ['a', 'c']:

            if f'frequency_{channel}' not in data.columns:
                continue


            for g in gridlist:
                f = data.loc[data['grid'] == g]
                f = f[f[f'frequency_{channel}'] > freqrange[0]]
                f = f[f[f'frequency_{channel}'] < freqrange[1]]

                if f.empty:
                    temp = pd.DataFrame()
                    temp["channel"] = [channel]
                    temp['grid'] = [g]
                    temp["file"] = [experiment]
                    temp["frequency"] = [np.nan]
                    temp["amplitude"] = [0]
                    # temp["pointspermicron2"] = [np.nan]
                    final = pd.concat([final, temp], axis=0)
                else:
                    # Autocorrelation
                    f2 = f.loc[f[f'periodicity_{channel}'].idxmax()].to_frame().transpose()
                    temp = pd.DataFrame()
                    temp["channel"] = [channel]
                    temp['grid'] = [g]
                    temp["file"] = [experiment]
                    temp["frequency"] = f2[f'frequency_{channel}'].values[0]
                    temp["amplitude"] = f2[f'periodicity_{channel}'].values[0]
                    # temp["pointspermicron2"] = f2[f'pointspermicron2_{channel}'].values[0]
                    final = pd.concat([final, temp], axis=0)

        if ('frequency_a' in data.columns) and ('frequency_c' in data.columns):
            for g in gridlist:
                f = data.loc[data['grid'] == g]
                f = f[f[f'frequency_a'] > freqrange[0]]
                f = f[f[f'frequency_a'] < freqrange[1]]
                f = f[f[f'frequency_c'] > freqrange[0]]
                f = f[f[f'frequency_c'] < freqrange[1]]
                f = f[f['periodicity_a'] > perlimit]
                f = f[f['periodicity_c'] > perlimit]
                if f.empty:
                    temp = pd.DataFrame()
                    temp['grid'] = [g]
                    temp["file"] = [experiment]
                    temp["frequency_a"] = [np.nan]
                    temp["frequency_c"] = [np.nan]
                    temp["amplitude_a"] = [np.nan]
                    temp["amplitude_c"] = [np.nan]
                    temp["amplitude_mean"] = [np.nan]
                    temp["cross-correlation_shift"] = [np.nan]
                    finalCC = pd.concat([finalCC, temp], axis=0)
                else:
                    # Crosscorrelation
                    f['amplitude_mean'] = f[['periodicity_a', 'periodicity_c']].mean(axis=1)
                    f2 = f.loc[f['amplitude_mean'].idxmax()].to_frame().transpose()
                    temp = pd.DataFrame()
                    temp['grid'] = [g]
                    temp["file"] = [experiment]
                    temp["frequency_a"] = f2['frequency_a'].iloc[0]
                    temp["frequency_c"] = f2['frequency_c'].iloc[0]
                    temp["amplitude_a"] = f2['periodicity_a'].iloc[0]
                    temp["amplitude_c"] = f2['periodicity_c'].iloc[0]
                    temp["amplitude_mean"] = f2['amplitude_mean'].iloc[0]
                    temp["cross-correlation_shift"] = f2['crosscorlag'].iloc[0]
                    finalCC = pd.concat([finalCC, temp], axis=0)





final.to_csv(os.path.join(root, f'{out_name}.csv'), sep=';')
if not finalCC.empty:
    finalCC.to_csv(os.path.join(root, f'{out_name}_CC.csv'), sep=';')

