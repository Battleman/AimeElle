# aprl-cs433-ml-project
## Quick summary
An important measure for environment is the air quality throughout time and locations. For that, we place a filter (weighted precisely beforehand) somewhere, let it rest for some time, then take it back and weighted again. The weight difference indicates the total quantity of air particles that were absorbed. Then, to obtain a precise distribution of composition, many expensive chemical treatments are applied to the filter (destructively).

This process, while being precise and efficient, is too expensive (requires complex machines) to be applied in less developed countries. But a new method is being developed, using infrared lasers. Before destruction with the chemical treatment, a laser is passed through the filter (non-destructively) and the spectrum is recorded. We also know the spectrum after going through a clean filter, in comparison.

The spectrum is slightly modified according to the content of the filter. Certain particles will modify the spectrum differently. Our goal here is to know if, given a certain spectrum, we can determine the content of the filter. If efficient, this method could be applied to a lot more areas worldwide and help improve air quality in less developed countries.

## Data
No data is present here, as we are tied to a NDA until the paper is published. If you believe you have permission to consult the data, you can access them [here](https://drive.google.com/drive/u/2/folders/1-2G_9KL-o6a38M7Tkj4irSj969DPj6Oa). This drive is the property of Prof. Satoshi Takahama. You must ask for permission to consult the data. Once this is done, put everything in the folder `data/`. Once this is done, you should have the following structure:
```
.
├── data/
│   ├── _gitkeep
│   ├── IMPROVE_2015_2nd-derivative_spectra_cs433.csv
│   ├── IMPROVE_2015_measures_cs433.csv
│   ├── IMPROVE_2015_raw_spectra_cs433.csv
│   └── IMPROVE_2015_train_test_split_cs433.csv
├── data-exploration.ipynb
├── figures/
├── loss.py
├── main.py
├── Modeling.ipynb
├── models.py
├── preprocessing.py
├── project-presentation.html
├── project-presentation.ipynb
├── README.md
├── utils.py
└── visualization.py
```

## Requirements
### Software
To run the code, ensure you have the data and the following python**3** packages:
* Numpy
* SKLearn
* Pandas
* Matplotlib

### Hardware
Even though we did our best to optimize the running, this project is still demanding in memory and computational power. With a total of
## Howto
There are two ways to run the code. If you want to enjoy some comments and visualization, the best way is to run the notebook `main.ipynb`. If you just want to generate the plots (in `./figure/`) and the results (in `results.csv`), you can simply launch `main.py`.

It is advised to use the notebook, 