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
python my-python-modules/manage_ssd_train.py
```

## Submitting Python Application at LoveLace environment 

Version of CUDA module to load: 
- module load cuda/11.5.0-intel-2022.0.1

```
qsub wm-model-ssd.script
```
```
qstat -u rubenscp 
```
```
qstat -q umagpu 
```
```
qstat -f 'job_id' 
```

The results of job execution can be visualizedat some files as:
- errors
- output 


CUDA 
old - module load cudnn/8.2.0.53-11.3-gcc-9.3.0
new - module load cuda/12.0.0-gcc-8.5.0 
