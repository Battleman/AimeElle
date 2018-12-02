# aprl-cs433-ml-project
This directory contains all code relative to the project with the APRL from EPFL. No data is present here, as we are tied to a NDA until the paper is published.

## Quick summary
An important measure for environment is the air quality throughout time and locations. For that, we place a filter (weighted precisely beforehand) somewhere, let it rest for some time, then take it back and weighted again. The weight difference indicates the total quantity of air particles that were absorbed. Then, to obtain a precise distribution of composition, many expensive chemical treatments are applied to the filter (destructively).

This process, while being precise and efficient, is too expensive (requires complex machines) to be applied in less developed countries. But a new method is being developed, using infrared lasers. Before destruction with the chemical treatment, a laser is passed through the filter (non-destructively) and the spectrum is recorded. We also know the spectrum after going through a clean filter, in comparison.

The spectrum is slightly modified according to the content of the filter. Certain particles will modify the spectrum differently. Our goal here is to know if, given a certain spectrum, we can determine the content of the filter. If efficient, this method could be applied to a lot more areas worldwide and help improve air quality in less developed countries.

## Method

## Requirements
The directory `data` is absent from the repository, as they were not made public yet. Once the research paper will be out, data may be made public (probably not on this repository though, as they are too heavy for a github).

To run the code, ensure you have the data and the following python packages:
* Numpy
* ScikitLearn
* Pandas