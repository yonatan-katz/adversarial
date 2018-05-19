Download dataset from: https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set/data/
Copy categories.csv and images.csv into ./data
Copy images in to ./data/images

In order to test the framework run: python evaluation.py
Is assumed using Python 3.6

install conda environment:
conda env create -f env.yaml
activate env:
source activate py36

install:
pip install -e git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
pip install tensorflow
