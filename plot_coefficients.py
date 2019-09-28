import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pandas as pd

import os
import sys


def plot_coefficients(df_coefficients, aoa=None):
    fig, axis = plt.subplots(3, 1, sharex=True, figsize=(14, 12))
    for ax, force_label in zip(axis, ["cm", "cl", "cd"]):
        df_coefficients.plot(x="time", y=force_label, style="-.", grid=True, ax=ax)
    if aoa:
        fig.suptitle("aoa = {}".format(aoa))
    return fig


def get_coefficients_df(case_dir):
    coeficient_filename = os.path.join(case_dir, "postProcessing", "forceCoeffs1", "0",
                                       "forceCoeffs.dat")
    return pd.read_csv(coeficient_filename, sep="\s+", skiprows=10,
                       names=["time", "cm", "cd", "cl"],
                       usecols=[0, 1, 2, 3])


def run_fft(t, signal):
    "Returns dataframe with sorted period&intensity"
    size_signal = int(len(t)/2)
    sampling_period =  t[1]-t[0]
    FFT = np.real(scipy.fft(signal))[:size_signal]
    freqs = scipy.fftpack.fftfreq(signal.size, sampling_period)[:size_signal]
    _, FFT_sorted, freq_sorted = zip(*sorted(zip(abs(FFT), FFT, freqs), reverse=True))
    return pd.DataFrame(data=zip(1/np.array(freq_sorted), np.array(FFT_sorted)/len(t)), columns=["period", "intensity"])


def get_aerodynamic_coefficients(df_coefficients, method='mean', skip_convergence=300, dirname=None):
    if method == "mean" and skip_convergence < df_coefficients.shape[0]:
        return df_coefficients[["cl", "cd", "cm"]].iloc[skip_convergence:].mean()
    elif method == "fft" and skip_convergence < df_coefficients.shape[0]:
        starting_time = 200
        total_time = 1500
        t = df_coefficients.index.values[starting_time:total_time]
        data = {}
        for coefficient in ["cm", "cd", "cl"]:
            signal = df_coefficients[coefficient].values[starting_time:total_time]
            df_FFT = run_fft(t, signal)
            if dirname:
                filename = os.path.join(dirname, "postProcessing", "forceCoeffs1", "0",
                                         "{}_fft.csv".format(coefficient))
                with open(filename, "w") as fid:
                    df_FFT.to_csv(fid, sep=" ", index=False)
            data[coefficient] = df_FFT.iloc[0]["intensity"]
        return pd.Series(data=data)
    else:
        return df_coefficients[["cl", "cd", "cm"]].iloc[-1]


if __name__ == "__main__":
    casedir = sys.argv[1]
    caselist = [case for case in os.listdir(casedir) if "case_aoa" in case]
    df_coefficients_all = pd.DataFrame(columns=["aoa", "cl", "cd", "cm"])
    for case in caselist:
        aoa = float(case.replace("case_aoa", ""))
        try:
            dirname = os.path.join(casedir, case)
            df_coefficients = get_coefficients_df(dirname)
            #fig = plot_coefficients(df_coefficients)
            #fig.savefig(os.path.join(dirname, "aerodinamic_coefficients.png"))

            results = get_aerodynamic_coefficients(df_coefficients, method="fft", dirname=dirname)
            results = results.append(pd.Series({"aoa": aoa}))
            print("results = \n{}".format(results))
            df_coefficients_all = df_coefficients_all.append(results, ignore_index=True)
        except Exception as e:
            print(e)
            print("Could not find results for aoa = {} deg".format(aoa))
    df_coefficients_all.sort_values(by="aoa", axis=0, inplace=True, ascending=True)
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    df_coefficients_all.plot(y="cl", x="aoa", ax=ax[0], label="cl", grid=True, style="-*")
    df_coefficients_all.plot(y="cm", x="aoa", ax=ax[1], label="cm (0.25c)", grid=True, style="-*")
    df_coefficients_all.plot(x="cl", y="cd", ax=ax[2], label="cd", grid=True, style="-*")
    plt.legend()
    plt.show()

    output_coefficients_filename = os.path.join(casedir, "coefficients.txt")
    with open(output_coefficients_filename, 'w') as fid:
        fid.write(df_coefficients_all.to_csv(index=False))



