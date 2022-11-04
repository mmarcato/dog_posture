# Posture Algorithm
The main goal of this machine learning project is to predict canine body posture based on Inertial data. 

In order to achieve that, the inertial data were colleted during a video recorded behaviour test. The videos were used for data annotation to create the labels for the dataset. 
- Inertial data acquired by three Actigraph IMUs (Inertial Measurement Units);
- Posture labels timestamped considering 5 classes of postures performed by canines;
- Type labels timestamped consider the 2 types of postures (Static, Dynamic);

# Folders
This section explains the structure of this folder, outlines the purpose of some scripts and explaints the content or origin of some files as appropriate.

## models
This folder contains the scripts developed to create all models that were experimented with. The most important folder is final as it contains the files for classifiers 1, 2 and 3 mentioned in the manuscript. 

## results
Contains image files (.png) with confusion matrices created using data files (.csv) with test and golden set predictions classifiers 1, 2 and 3 which were presetend in the manuscript.

## scr
Folder with the source code created 
- modules: helper modules used to import data, evaluate models, learn using pipelines, process raw datasets to create processed dataset, etc. These files store code/functions to help in all stages of the machine learning algorithm;
- data: scripts for creating df_raw.csv dataframe by combining data from several IMUs and the posture labels. More details are provided in the Data Workflow section;
- features: scripts for creating a processed dataframe by extracting features from raw data using different methods;

# Dataset

The data folder which contains dataframes related project is not available on GitHub due to file size issues. The following two raw datasets were made publicly available on Mendeley Data, repository name [Inertial sensor dataset for Dog Posture Recognition
](https://data.mendeley.com/v1/datasets/mpph6bmn7g/draft?a=94173ece-2a20-4d70-9ca3-f358d4703ecb):

- **df_dogs.csv**: demographic data for the dogs downloaded from 'Data Collection - Dogs.csv'
</details>

- **df_raw.csv**: contains raw IMU data with the position and type labels. This file is created by src/data/raw.py, considering both timestamps and raw IMU datasets. Different versions were created as more data were labeled, use the latest one. The Data Workflow section explains how this was created.

Here is an overview of the workflow used to create df_raw.csv by combining information from several raw files (3 x IMUs, Summary and Timestamps):

<details><summary>  Step 1: Creates IMU raw files in day folder *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).csv*. </summary>
    ActiLife software is used to download the data from the IMUs and export it to .csv files. The raw data acquired in one data collection day are recorded by three Actigraph IMUs placed on different body parts (back, chest, neck). ActiLife software is used to download the data from each of those, resulting in three raw ActiLife files *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).gt3x and agd* file formats. ActiLife Software is used to export those three raw ActiLife .gt3x files into IMU .csv files. Each of three files contain 3-axial accelerometer, gyroscope, magnetometer data for all the dogs tested in a given day.</details>

<details><summary>  Step 2: Creates Actigraph raw files in dog folder *Subjects/DogName/X_Actigraph/(Back, Chest, Neck).csv* where X is Data Collection Number. </summary>
    Python script in path *scr/data/imu_select.py*. (1) Imports *Data Collection - Summary.csv* to extract each dog's behaviour test start and finish datetime data. (2) Imports & selects data considering the dog's data collection time from the three raw IMU located at *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).csv* files. (3) Saves selected data in the dog's folder. Path format: *Subjects/DogName/X_Actigraph* where X is Data Collection Number.</details>

<details><summary>  Step 3: Creates posture timestamps file in dog folder *Subjects/DogName/X_Timestamps.csv* where X is Data Collection Number. </summary>
    Excel macro saves .csv time to dog folder from postures labels in the Timestamp excel file.</details>

<details><summary>  Step 4: Creates df_raw.csv. </summary>
    Python script in path *scr/data/raw_create.py*. (1) Imports Raw ActiLife data in *Subjects/DogName/X_Actigraph/(Back, Chest, Neck).csv* and Timestamps is *Subjects/DogName/X_Timestamps.csv*. (2) Combines them to create df_raws. (3) Imports *Data Collection - Dogs* data for demographical information and data exploration.</details>

Note: Folders named *Actigraph* and *Subjects* are outside this repository.
