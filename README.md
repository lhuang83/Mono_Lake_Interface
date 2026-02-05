# README

## Brief Overview

This is a quick reference to understand the files used in the **UCLA Mono Lake Model (UCLA-MLM) User Interface**.

### File Descriptions

- **UCLA_MLM_User_Interface.ipynb**: This is the primary user interface, which allows users to interactively set-up and simulate how Mono Lake Water Level responds to a different export criteria and climate scenarios

- **Mono_Lake_Interface_lib_v4.py**: This contains the bulk of python code functions used to represent the Mono Lake Water Budget (MLWB) and used to create figures in the user interface (this file is loaded into UCLA_MLM_User_Interface.ipynb) 

- **data.tar**: This contains the climate projections across Mono Basin that are required as critical components for the Mono Lake Water Budget (e.g. precipitation on Mono Lake). It is unpacked in Step 0 of the notebook.

- **Mono_Lake_Area_Storage_Elev.txt**: This contains the relationship between Mono Lake storage, water level, and surface area

- **details_for_model.csv**: This contains a few fine-tuned parameters that are used by the UCLA-MLM

## Instructions for how to run the UCLA-MLM are provided in the User Interface (UCLA_MLM_User_Interface.ipynb)
## Background information on Mono Lake and further details associated with the UCLA-MLM can be found in the accompanying report delivered to the SWRCB
