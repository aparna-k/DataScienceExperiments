Helpers for https://www.kaggle.com/c/avito-demand-prediction

#### Translate titles and Descriptions

`translate_avito_title_descriptions`

**Usage**

```
python translate_avito_title_descriptions.py --help
usage: translate_avito_title_descriptions.py [-h] -f F [-pre [PRE]]
                                             [-title-only [TITLE_ONLY]]
                                             [-desc-only [DESC_ONLY]]

optional arguments:
  -h, --help            show this help message and exit
  -f F                  input file path
  -pre [PRE]            prefix for output files
  -title-only [TITLE_ONLY]
                        pass yes if only title has to be translated
  -desc-only [DESC_ONLY]
                        pass yes if only description has to be translated
```

**Example:**

Take a random slice of data from `train.csv` in the original dataset. Ex: `small_sample.csv`

Run the script as follows:

```
python translate_avito_title_descriptions.py -f ./input/small_sample.csv
```

**Optional Arguments:**

`-pre` - prefix for output files

`-title-only` - pass yes if only title has to be translated

`desc_only` - pass yes if only description has to be translated