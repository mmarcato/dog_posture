# Posture_Recognition
* Inputs: 
    * **dir_base**: directory containing data organised in folders by Dog_Name

* Script:
    * dir_base//DC_Actigraph.csv: It will only import Actigraph data
    * dir_base//DC_Timestamps.csv: File containing the episode and positions. 
        * It is automatically created using file. Path 'Study//TimeStamps//Template.csv'


# Split_File 
* Inputs:
    * **dir_base**: file path to 'Data Collection - Summary.csv'
    * **dir_raw**: directory containing Actigraph data organised in folders by Date 
    * **dir_new**: directory to save new filtered Actigraph data 

* Script: 
    * From Data Collection: reads Summary as csv files, import 'BT Start' and 'BT Finish' for each dog
    * From Raw: create new df with Actigraph data within the 'BT Start' and 'BT Finish' times
    * Save new df as csv in dir_new ('Subject/DC_Actigraph/') and appropriate name ('YYYY-MM-DD_BodyPart')
