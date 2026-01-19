####################################################################
#
# File Name :   liver_service.py
# Description : Dummy ML prediction service for liver disease
# Author      : Pradhumnya Changdev Kalsait
# Date        : 17/01/26
#
####################################################################

"""
################################################################
#
# Function Name : predict_liver_disease
# Description   : Simulates liver disease and criticality prediction
# Author        : Pradhumnya Changdev Kalsait
# Date          : 17/01/26
# Prototype     : dict predict_liver_disease(dict)
# Input Output  : (1 input, 1 output)
#
################################################################
"""
def predict_liver_disease(input_data):

    return {
        "organ": "LIVER",
        "disease": "Cirrhosis",
        "criticality": "MEDIUM",
        "decision": "MEDICATION & MONITORING"
    }
