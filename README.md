# AutoMPP
Automated tool for molecular property prediction

## Environment Requirements
- Python == 3.9
- AutoGluon == 1.1.1
- RDKit

**1. Create a new conda environment**

        conda create -n AutoMPP python=3.9

**2. Activate the environment**

        conda activate AutoMPP

**3. Install core dependencies**

        pip3 install autogluon==1.1.1

        pip3 install rdkit

## Training

**1. Split the data into training and test sets
        python train.py task_name
