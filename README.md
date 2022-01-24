# Posture Algorithm
The main goal of this machine learning project is to predict canine body posture based on Inertial data.

In order to achieve that, the following data are colleted during a video recorded behaviour test are used.
* Inertial data acquired by three Actigraph IMUs (Inertial Measurement Units)  
* Posture labels timestamped considering 9 classes of postures performed by canines
* Type labels timestamped consider the 2 types of postures (Static, Dynamic)

The data workflow overview: 

    Actigraph Raw Files in day folder -> Actigraph/YYYY-MM-DD/(Back, Chest, Neck).csv
    
    Start and Finish Timestamps for each dog -> Data Collection - Summary.csv

    Actigraph Raw Files in dog folder considering Start & Finish Timestamps -> Subjects/DogName/X_Actigraph/(Back, Chest, Neck).csv where X is Data Collection Number

    Timestamps for Postures file -> Subjects/DogName/X_Timestamps.csv where X is Data Collection Number

    Actigraph Dog Body Parts Combined + Timestamps for Postures for all dogs -> df_raw.csv

The data workflow detailed description is as follows:

* Step 1: The raw data acquired in one data collection day are recorded by three Actigraph IMUs placed on different body parts (back, chest, neck). ActiLife software is used to download the data from each of those, resulting in three raw ActiLife files (back, chest, neck). Path format: *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).gt3x and agd* file formats.

* Step 2: ActiLife Software is used to export those three raw ActiLife files into IMU .csv files. Each of resulting files (back, chest, neck) contain 3-axial Accelerometer, Gyroscope, Magnetometer data for all the dogs tested in a given day. Path format: *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).csv* files.

* Step 3: Python script in path *scr/data/imu_select.py*. (1) Imports Data Collection - Summary.csv and takes each dog's behaviour test start and finish datetime data. (2) Imports & selects data considering the  dog's data collection time from the three raw IMU located at *Actigraph/YYYY-MM-DD/(Back, Chest, Neck).csv* files. (3) Saves selected data in the dog's folder. Path format: *Subjects/DogName/X_Actigraph* where X is Data Collection Number.

* Step 4: Labels were created for the body postures performed by the subjects and added to a Timestamp excel file. There is a macro to save each of the tabs as a csv file in the dog's folder.

* Step 5: Python script in path *scr/data/raw_create.py*. (1) Imports Raw ActiLife data in *Subjects/DogName/X_Actigraph/(Back, Chest, Neck).csv* and Timestamps is *Subjects/DogName/X_Timestamps.csv*. (2) Combines them to create df_raws. (3) Imports Data Collection - Dogs data to for dataset statistics and data exploration.



# WS: Dog Posture
This file explains the structure of this folder and outlines the content and source of each file as appropriate.

## data
Contains dataframes related project. Not available on GitHub due to file size issues.
<details><summary> Raw: Contains unprocessed dataframes </summary>

* **df_raw.csv**: contains raw IMU data for timestamps labeled by positions. This file is created by src/data/raw.py, considering Timestamps and raw IMU data.

* **df_dogs.csv**: demographic data for the dogs downloaded from 'Data Collection - Dogs.csv'
</details>

<details><summary> Processed: dataframe.cvs and dataframe.log</summary> <p>  
The folder structure contains .csv files and an associated .log file with the hyperparameters used to create them.
</details>

## models

## results

## scr
modules: local modules created to store functions to help in all stages of the machine learning algorithm


* Inputs: 
    * **dir_base**: directory containing data organised in folders by Dog_Name

* Script:
    * dir_base//DC_Actigraph//.csv: It will only import Actigraph data
    * dir_base//DC_Timestamps.csv: File containing the episode and positions. 
        * It is automatically created using file. Path 'Study//TimeStamps//Template.csv'



