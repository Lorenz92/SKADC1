# SKADC1

This repository contains solutions to the [SKADC1](http://www.skatelescope.org/news/ska-launches-science-data-challenge) challenge. In particular we focused on the 560MHz-1000h high S/N sky image, that contains more than 19.000 radio sources.

The original dataset consists in the following columns:


| Column | ID | unit of measure | Description |
| :-: | :- | :-: | :- |
| 1 | ID | none | Source ID |
| 2 | RA (core) | degs | Right ascension of the source core |
| 3 | DEC (core) | degs | DEcination of the source core |
| 4 | RA (centroid) | degs | Right ascension of the source centroid |
| 5 | DEC (centroid) | degs | Declination of the source centroid |
| 6 | FLUX | Jy | integrated flux density |
| 7 | Core frac | none | integrated flux density of core/total |
| 8 | BMAJ | arcsec | major axis dimension |
| 9 | BMIN | arcsec | minor axis dimension |
| 10 | PA | degs | PA (measured clockwise from the longitude-wise direction) |
| 11 | SIZE | none | 1,2,3 for LAS, Gaussian, Exponential |
| 12 | CLASS | none | 1,2,3 for SS-AGNs, FS-AGNs,SFGs |
| 13 | SELECTION | none | 0,1 to record that the source has not/has been injected in the simulated map due to noise level |
| 14 | x | none | pixel x coordinate of the centroid, starting from 0 |
| 15 | y | none | pixel y coordinate of the centroid,starting from 0 |

## Installation

In order to install all the dependencies required by the project, you can use `pip`: make sure that you have `Python 3.8` installed on your system and run

```bash
python3 -m venv skadc1
source skadc1/bin/activate
pip3 install -r init/requirements.txt

```

### Important!

In order to have a fully working code you should manually implement the following fix to your scikit-learn package: please refer to [this](https://github.com/scikit-learn/scikit-learn/issues/8245#issuecomment-276682354) GitHub issue.



## Execution

Training and evaluation of models is managed through a jupyter notebook where you can select which model to train and which hyperparameters (some of them) to use. The model choice is managed through a variable called `backbone`, where possible values are: `baseline_16, baseline_44, vgg16`.

### Training

The training phase is carried out in a dedicated section in the notebook.

Training and evaluation metrics, along with model checkpoints and results, are directly saved on a local folder under `model/<backbone>/` directory.

During the training phase two kind of weights are saved:

- `loss_<counter>_frcnn_<backbone>.hd5`
- `map_<counter>_frcnn_<backbone>.hd5`

where the firsts are saved eny time the loss decrease from the best value so far, and the seconds are saved any time mAP increase with respect to the best value so far.

### Evaluation

For the model avaluation there is a dedicated section too.

Before running the evaluation loop you should compile your chosen model and load weights to use during evaluation. For loading weights use `checkpoint` variable. It is also possible to select a different threshold for evaluation metrics by changing the `metric_threshold`variable in the`evaluate_model` function.

The evaluation loop saves patches used for the evaluation alogside the detection in `data/prediction/patches/`.

For a qualitative evaluation there is the `print_img` function that print the given image and draw ground truth bounding boxes alongside predicted bounding boxes (if any or if given).
