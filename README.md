# White Mold Pre-Processing Image Dataset

### Institute of Computing (IC) at University of Campinas (Unicamp)

### Postgraduate Program in Computer Science

### Team

* Rubens de Castro Pereira - student at IC-Unicamp
* Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
* Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
* Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans

### Main purpose

This Python project aims to support the pre-processing of the images used to train and inference the neural networks models evaluated.

# Installing Python Virtual Environment
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

# Running Python Application 

```
access specific folder 'wm-model-ssd'
```
```
execute command: python my-python-modules/manage_ssd_train.py
```