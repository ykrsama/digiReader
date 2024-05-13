# Digital Board Reader

## Setup Environment

```bash
my_environment=digireader
conda config --set channel_priority strict
conda env create --name $my_environment --file environment.yml
conda activate $my_environment
```

## Exampmle Usage

**Draw first several waves individually**

```bash
python3 reader.py <filenames.bin> --wave 3 --algo denoise
```

**Draw waveform density plot**

when `--wave` >= 10 :

```bash
python3 reader.py <filenames.bin> --wave 100 --algo denoise
```

**Read a slice of binary file**

```bash
python3 reader.py <filenames.bin> --id 50001 10000
# output id from 50001 to 60000
```

process entire binary file when whithout `--id` .

`--wave`/`--noroot` option will suppress root ouptut.

**Apply denoise and gaussian mixture baseline**

```bash
python3 reader.py <filenames.bin> --algo denoise gmm --wave 3
# or without --wave to output root file
```

## Notes

Constant baselines are used to correct the integration. The integrated value is store in root file branch: `net_signal_median` (`net_signal_gmm`, `net_signal_landau`)

**Baeline algorithm:**

There are three constant baseline algorithms, can be turned on by user in `--algo`

The default baseline is calculated by median number of the data.

If `gmm`/`landau` for baselines calculated by the Gaussian Mixture Model / Landau distribution fit.

**Deonise algorithm:**

If add `denoise` in `--algo`, the baseline will be calculated by the denoised waveform.

The deonise algorithm is Savitzky-Golay filter.

The window length and polyorder can be optimized using `optimize_denoise_savgol.py`
