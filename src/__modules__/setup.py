''' 
    This file contains parameters for the posture recognition algorithm
'''
from datetime import timedelta
import logging
# ------------------------------------------------------------------------- #
#                               Setup Logger                                #    
# ------------------------------------------------------------------------- #

def log(name, log_file = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\main.log', level = logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(name)s \n%(message)s')
    handler = logging.FileHandler(log_file)       
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
    

# ------------------------------------------------------------------------- #
#                            Feature Engineering                            #    
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                             Machine Learning                              #    
# ------------------------------------------------------------------------- #

df_fname = 'df_11'
run = 'GS-GB-df_32'
