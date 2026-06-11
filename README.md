# AutoMPP
Automated tool for molecular property prediction

<img width="1956" height="1108" alt="image" src="https://github.com/user-attachments/assets/97d26616-a3ab-4754-91ae-cb33e7daea27" />


## Environment Requirements
- Python == 3.9
- AutoGluon == 1.1.1
- RDKit

**1. Create a new conda environment.**

        conda create -n AutoMPP python=3.9

**2. Activate the environment.**

        conda activate AutoMPP

**3. Install core dependencies.**

        pip3 install autogluon==1.1.1

        pip3 install rdkit

## Training

**1. Divide the molecular property data in the data folder into training sets and test sets.**

**2. train the model.**

        python train.py task_name
