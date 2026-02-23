# README

## Brief Overview

This is a quick reference to understand the files used in the **UCLA Mono Lake Model (UCLA-MLM) User Interface**.

### File Descriptions

- **UCLA_MLM_User_Interface.ipynb**: This is the primary user interface, which allows users to interactively set-up and simulate how Mono Lake Water Level responds to a different export criteria and climate scenarios

- **Mono_Lake_Interface_lib_v4.py**: This contains the bulk of python code functions used to represent the Mono Lake Water Budget (MLWB) and used to create figures in the user interface (this file is loaded into UCLA_MLM_User_Interface.ipynb) 

- **data.tar**: This contains the climate projections across Mono Basin that are required as critical components for the Mono Lake Water Budget (e.g. precipitation on Mono Lake). It is unpacked in Step 0 of the notebook.

- **Mono_Lake_Area_Storage_Elev.txt**: This contains the relationship between Mono Lake storage, water level, and surface area

- **details_for_model.csv**: This contains a few fine-tuned parameters that are used by the UCLA-MLM

### Instructions for how to run the UCLA-MLM are provided in the User Interface (UCLA_MLM_User_Interface.ipynb)

# Overview of Shared Socioecononomic Pathways (SSPs)

CMIP6 simulates GCMs under historical emissions and different greenhouse gas emission trajectories, known as Shared Socioeconomic Pathways (SSPs). Different SSPs exist to allow for assessments of how varying levels of emissions related to different societal choices may influence future climate change.

The UCLA-MLM focuses on the three SSPs available from CA5: SSP2-4.5, SSP3-7.0, and SSP5-8.5. These scenarios were selected because they span a range of plausible future emissions pathways, from moderate (SSP2-4.5) to very high (SSP5-8.5), relative to historical levels (Fifth National Climate Assessment, 2023). There is substantial uncertainty associated with which emission scenario will be realized; however, SSP2-4.5 and SSP3-7.0 are generally considered more likely than SSP5-8.5 (Huard, 2022).

- SSP2-4.5 includes moderate efforts to mitigate climate change and reflects a world with gradual progress toward sustainability.
- SSP3-7.0 is often considered an intermediate-high or "business as usual" pathway, where global efforts to reduce greenhouse gas emissions are limited.
- SSP5-8.5 is often referred to as the "worst-case scenario" and assumes rapid economic growth fueled by intensive fossil fuel use.

# Overview of Transition Export Criteria and Post-Transition Export Criteria

![Alt text](images\Table%204-1%20Overview%20of%20Transition%20and%20Post-Transition%20Export%20Criteria.png)

## Transition Export Criteria
![Alt text](images/Transition%20Export%20Criteria%20A1%20to%20A8.png)
![Alt text](images/Transition%20Export%20Criteria%20A9%20to%20A10.png)

## Post-Transition Export Criteria

![Alt text](images/Post-Transition%20Export%20Criteria.png)

## More background information on Mono Lake and further details associated with the UCLA-MLM can be found in the accompanying report delivered to the SWRCB