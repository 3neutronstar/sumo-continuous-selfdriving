# flow_edgecloud
Reinforcement Learning based autonomous driving system connected with edge-cloud

### Requirement(Installment)
- Ubuntu 18.04LTS is recommended. (Window OS is not supported.)
- anaconda : https://anaconda.com/
- flow-project : https://github.com/flow-project/flow
- ray-project(rllib) : https://github.com/ray-project/ray (need at least 1.1.0 is needed)
- pytorch : https://pytorch.org/


### How to Download Requirement
#### Anaconda(Python3) installation:
- Prerequisites
```shell script
    sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
- Installation(for x86 Systems)
In your browser, download the Anaconda installer for Linux (from https://anaconda.com/ ), and unzip the file. 
``` shell script
bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
We recomment you to running conda init 'yes'.<br/>
After installation is done, close and open your terminal again.<br/>


#### Flow installation
Download Flow github repository.
```shell script
    git clone https://github.com/flow-project/flow.git
    cd flow
``` 
We create a conda environment and installing Flow and its dependencies within the enivronment.
```shell script
    conda env create -f environment.yml
    conda activate flow
    python setup.py develop
```
For Ubuntu 18.04: This command will install the SUMO for simulation.<br/>
```shell script
scripts/setup_sumo_ubuntu1804.sh
```
For checking the SUMO installation,
```shell script
    which sumo
    sumo --version
    sumo-gui
```
(if SUMO is installed, pop-up window of simulation is opened)
- Testing your SUMO and Flow installation
```shell script
    conda activate flow
    python simulate.py ring
```

#### Torch installation (Pytorch)
You should install at least 1.6.0 version of torch.(torchvision: 0.7.0)
```shell script
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

#### Visualizing with Tensorboard
If tensorboard is not installed, you can install with pip, by following command `pip install tensorboardx`.


### Contributor