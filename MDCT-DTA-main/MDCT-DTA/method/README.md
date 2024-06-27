### Intro

-----------

This is the code that is used as reference material, which contains the dataset, body model, validation metrics and other materials that you can find in the corresponding folders.  

The corresponding code runtime dependencies are given in the requirements.txt file as a reference.  

It is recommended to use the commonly used SMILES/FASTA data and Graph Transform processing methods for the corresponding type of datasets, and you can use MSE loss for model training.

### Drug-target binding affinity prediction model based on multi-scale diffusion and interactive learning

-----------

Drug-target interactions (DTIs) play a key role in drug discovery and development as they are critical in understanding the complex mechanisms of underlying drugs and their corresponding targets. Unfortunately, some studies do not fully recognize the significant contribution of graph structure to drug feature extraction, thereby neglecting the extraction of local and global structures inherent in molecular graphs. Furthermore, most existing drug-target binding affinity (DTA) prediction models ignore or simplify complex interaction mechanisms, thus compromising the predictive power of the models. Therefore, this paper proposes a novel DTA prediction model utilizing multi-scale diffusion and interactive learning (MDCT-DTA).  

#### Datasets

----------

The datasets used in this paper are KIBA，Davis, Metz and BindingDB. The way to obtain the above datasets is given in the data file.

#### Requirement

The other libraries are listed in the requirements file.

#### Cite

----------------------------------------------

