import os
from os import path
BASE_DIR=path.abspath(path.dirname(path.dirname(__file__)))
#envirmonent settings with google cloud
AIP_MODEL_DIR=os.environ.get("AIP_MODEL_DIR")
BUCKET_NAME=os.environ.get("CLOUD_BUCKET","imdbml")
PROJECT_ID=os.environ.get("CLOUD_ML_PROJECT_ID","mlprac-321407")
CLOUD_TRAINDATA_PATH=os.environ.get("CLOUD_TRAINDATA_PATH")
CLOUD_TESTDATA_PATH=os.environ.get("CLOUD_TESTDATA_PATH")
CLOUD_TOKENIZER_PATH=os.environ.get("CLOUD_TOKENIZER_PATH")

#training configs
MAX_VOCAB_SIZE=25000
MIN_FREQ=2
SPECIAL_TOKENS=["[UNK]","[PAD]"]
ENTITY_MAPPINGS={"O":0,"CARDINAL_B":1,"CARDINAL_I":2,"DATE_B":3,"DATE_I":4,"EVENT_B":5,"EVENT_I":6,"FAC_B":7,
                 "FAC_I":8,"GPE_B":9,"GPE_I":10,"LANGUAGE_B":11,"LANGUAGE_I":12,"LAW_B":13,"LAW_I":14,"LOC_B":15,
                 "LOC_I":16,"MONEY_B":17,"MONEY_I":18,"NORP_B":19,"NORP_I":20,"ORDINAL_B":21,"ORDINAL_I":22,"ORG_B":23,
                 "ORG_I":24,"PERCENT_B":25, "PERCENT_I":26,"PERSON_B":27, "PERSON_I":28,"PRODUCT_B":29,"PRODUCT_I":30,
                 "QUANTITY_B":31,"QUANTITY_I":32,"TIME_B":33,"TIME_I":34,"WORK_OF_ART_B":35,"WORK_OF_ART_I":36}
LABEL_MAPPING={"neg":0,"pos":1}