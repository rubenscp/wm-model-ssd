# White Mold Single Shot Detector (SSD) Model 

### Institute of Computing (IC) at University of Campinas (Unicamp)

### Postgraduate Program in Computer Science

### Team

* Rubens de Castro Pereira - student at IC-Unicamp
* Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
* Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
* Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans

### Main purpose

This Python project aims to train and inference the Single Shot Detector (SSD) Model in the image dataset of white mold disease and its stages.

## Installing Python Virtual Environment
```
module load python/3.10.10-gcc-9.4.0
```
```
pip install --user virtualenv
```
```
virtualenv -p python3.10 venv-wm-model-ssd
```
```
source venv-wm-model-ssd/bin/activate
```
```
pip install -r requirements.txt
```

## Running Python Application 

```
access specific folder 'wm-model-ssd'
```
```
execute command: python my-python-modules/manage_ssd_train.py
```