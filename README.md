## Set up environment
We recommend you to set up a conda environment for packages used in this homework.
```
conda create -n hw4-part-1-nlp python=3.9
conda activate hw4-part-1-nlp
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk
```
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> exit()
```

Initial Testing: python3 main.py --train --eval --debug_train


Configure for google colab:

transformers>=4.30.0
datasets==2.9.0
torch>=2.0.0
evaluate==0.4.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk==3.8.1
pyarrow>=10.0.0,<15.0.0
fsspec<2023.10.0