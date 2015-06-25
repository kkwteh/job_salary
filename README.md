[Theory slides](https://docs.google.com/presentation/d/1YTRygbuEfKUJ7UnH-R6WRZvvVWNAe3uaARHf0ajko-g/edit?usp=sharing) by Katherine Luna
================

Setup instructions
==================
* Download training data at https://s3.amazonaws.com/stylometry/job_salary_data.zip
* Move job_salary_data.zip to the root directory and unzip
* Install Miniconda for Python 2.7. See installation instructions at http://conda.pydata.org/docs/install/quick.html
* `$conda create --name opendoor --file conda-requirements.txt`
* `$source activate opendoor`
* To train model run `$python model_training.py`, or run all the cells in `model_training.ipynb` in an IPython Notebook
* To run the web app locally, `$python job_salary.py`
* To send a request to the local web app, `$python client.py`
