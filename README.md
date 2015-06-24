Setup instructions
==================
* Download training data at https://s3.amazonaws.com/stylometry/Data.zip
* Install Miniconda. See installation instructions at http://conda.pydata.org/docs/install/quick.html
* `$conda create --name opendoor --file conda-requirements.txt`
* `$source activate opendoor`
* To train model run `$python job_salary_model_training.py`, or run all the cells in `job_salary_model_training.ipynb`
* To run the web app locally, `$python job_salary.py`
* To send a request to the local web app, `$python client.py`
