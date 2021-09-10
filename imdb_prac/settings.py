import os
from os import path
BASE_DIR=path.abspath(path.dirname(path.dirname(__file__)))
#envirmonent settings with google cloud
AIP_MODEL_DIR=os.environ.get("AIP_MODEL_DIR")
BUCKET_NAME=os.environ.get("CLOUD_BUCKET","imdbml")
PROJECT_ID=os.environ.get("CLOUD_ML_PROJECT_ID","mlprac-321407")
CLOUD_TRAINDATA_PATH=os.environ.get("CLOUD_TRAINDATA_PATH")
CLOUD_TESTDATA_PATH=os.environ.get("CLOUD_TESTDATA_PATH")

#training configs
MAX_VOCAB_SIZE=25000
MIN_FREQ=2
SPECIAL_TOKENS=["[UNK]","[PAD]"]
ENTITY_MAPPINGS={"O":0,"GPE_B":1,"GPE_I":2,"LANGUAGE_B":3,"LANGUAGE_I":4,"LAW_B":5,"LAW_I":6,"LOC_B":7,
                  "LOC_I":8,"MONEY_B":9,"MONEY_I":10,"NORP_B":11,"NORP_I":12,"ORDINAL_B":13,"ORDINAL_I":14,
                  "ORG_B":15,"ORG_I":16,"PERCENT_B":17, "PERCENT_I":18,"PERSON_B":19, "PERSON_I":20,"PRODUCT_B":21,
                 "PRODUCT_I":22,"QUANTITY_B":23,"QUANTITY_I":24,"TIME_B":25,"TIME_I":26,"WORK_OF_ART_B":27,
                 "WORK_OF_ART_I":21}
LABEL_MAPPING={"neg":0,"pos":1}