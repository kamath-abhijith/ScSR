# SPARSITY DRIVEN IMAGE SUPER RESOLUTION

This repository contains codes reproducing results from ["Image Super-Resolution using Sparse Representation"](https://ieeexplore.ieee.org/document/5466111), cite:

```shell
@ARTICLE{5466111,
  author={Yang, Jianchao and Wright, John and Huang, Thomas S. and Ma, Yi},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Super-Resolution Via Sparse Representation}, 
  year={2010},
  volume={19},
  number={11},
  pages={2861-2873},
  doi={10.1109/TIP.2010.2050625}}
```

## Documentation

This was a collaborative project done as part of E1 249 Digital Image Processing, Fall 2020 course at the Indian Institute of Science. Please refer to `docs/slides.pdf` for slides related to the complete project.

## Installation

Clone this repository and install the necessary dependencies
```shell
git clone https://github.com/kamath-abhijith/Vehicle_Tracking
conda create --name <env> --file requirements.txt
```

## Run
- Insert the validation images into "val_hr".
- Use the programme makedata.py to simulate low-resolution data.
- Select the dictionary parameters in main.py under "INITIALISE DICTIONARY",
- Please make sure the dictionary with the chosen parameters exists in "/data/dicts/",
- Else, use train_dict.py to train the dictionary.
- Select the reconstruction parameters in main.py under "INITIALISE PARAMETERS".
- Run main.py