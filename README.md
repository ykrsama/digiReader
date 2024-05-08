# Digital Board Reader

## Setup Environment

```bash
conda config --set channel_priority strict
conda env create --name <my-environment> --file environment.yml
conda activate <my-environment>
```

## Exampmle Usage

**Draw first 100 waves**

```bash
python3 read.py <filename.bin> --modes wave
```

**Read full binary file, and output root file**

```bash
python3 read.py <filename.bin> --modes root
```

**Read a slice of binary file**

`--buff` is optional, normally the program will find the header automatically.
Use this argument when the first found header is wrong.

```bash
python3 read.py <filename.bin> --modes root --id 50000 60000 --buff 0x2104
# id starting from 50000 to 60000, with length_buff = 0x21, offset_buff = 0x04
```

**Read a slice of binary file, apply denoise and Gaussian Mixture baseline**

```bash
python3 read.py <filename.bin> -m root denoise gmm -i 1 1000
# id starting from 1 to 1000
```

## Notes

Constant baselines are used to correct the integration. The integrated value is store in root file branch: `net_signal_median` (`net_signal_gmm`, `net_signal_landau`)

**Baeline algorithm:**

There are three constant baseline algorithms, can be turned on by user in `--modes`

The default baseline is calculated by median number of the data.

If `gmm`/`landau` for baselines calculated by the Gaussian Mixture Model / Landau distribution fit.

**Deonise algorithm:**

If add `denoise` in `--modes`, the baseline will be calculated by the denoised waveform.

The deonise algorithm is Savitzky-Golay filter.

The window length and polyorder can be optimized using `optimize_denoise_savgol.py`
