# Master Thesis Repository
This repository contains everything that was needed to write my master thesis. On a high level, I am working on improving upon the publication called "Random function priors for exchangeable arrays with applications to graphs and relational data" by Lloyd et al. ([2012](https://proceedings.nips.cc/paper/2012/hash/df6c9756b2334cc5008c115486124bfe-Abstract.html)). The original implementation of this model by the authors can be found [here](https://github.com/adiehl96/BasicRFM). The datasets that were used in their experimentes were archived [here](https://github.com/adiehl96/Network-Science-Datasets).

## Structure
The repository is divided into different directories that are self contained. As of right now, the following subdirectories exist:
* **dev**: This directory contains legacy implementations and minimum working examples of certain aspects of the RFM model described in the paper by Lloyd et al.
* **pyrfm**: A python implementation of the Lloyd et al. RFM model, written in an imperative programming style (as opposed to the object oriented style of the [original](https://github.com/adiehl96/BasicRFM)).

## Usage

Make sure to install the conda environment found in [environment.yml](./environment.yml). The conda command for installing an environment from a file is the following:
```
conda env create --file environment.yml
```
Further information on running a certain piece of software in this repository can be found in the subdirectories.

## Licence
Copyright (C) 2013, [Morten Mørup](http://www.mortenmorup.dk/)<br>
Copyright (C) 2012, 2013, [James Robert Lloyd](https://github.com/jamesrobertlloyd)<br>
Copyright (C) 2022, [Arne Diehl](https://github.com/adiehl96)


Everything in this repository is Licensed under the GNU General Public License v3.0. You can find the licence in the file [LICENCE](./LICENSE).