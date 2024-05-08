# Digital Board Reader

## Environment

```bash
conda config --set channel_priority strict
conda env create --name <my-environment> --file environment.yml
conda activate <my-environment>
```

## Draw first 100 waves

```bash
python3 reader.py <filename.bin> -m wave
```

## Output root file

```bash
python3 reader.py <filename.bin> -m root
```
