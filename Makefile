MODEL_QAG_BASE = VosLannack/QAG_ID_Generator_t5_base
MODEL_QG_BASE = VosLannack/QG_ID_Generator_t5_base
MODEL_QG_SMALL = VosLannack/QG_ID_Generator_t5_small
MODEL_DG_BASE = VosLannack/Distractor_all_t5-base
MODEL_DG_SMALL = VosLannack/Distractor_all_t5_small
MODEL_DG_1_SMALL = VosLannack/Distractor_1_t5-small

QG_PATH = VosLannack/squad_id_512
DG_1_PATH = VosLannack/race_id_uncombined
DG_PATH = VosLannack/race_id

QG_TYPE = qg 
QAG_TYPE = qag 
DG_TYPE = dg 
DG_1_TYPE = dg_1

ABS_PATH = /content/drive/MyDrive/Thesis

QG_SAVE_FILE_NAME_VAL = $(ABS_PATH)/QG_Pred_val.csv
QAG_SAVE_FILE_NAME_VAL = $(ABS_PATH)/QAG_Pred_val.csv
DG_SAVE_FILE_NAME_VAL = $(ABS_PATH)/DG_Pred_val.csv
DG_1_SAVE_FILE_NAME_VAL = $(ABS_PATH)/DG_1_Pred_val.csv


QG_SAVE_FILE_NAME_TEST = $(ABS_PATH)/QG_Pred_test.csv
QAG_SAVE_FILE_NAME_TEST = $(ABS_PATH)/QAG_Pred_test.csv
DG_SAVE_FILE_NAME_TEST = $(ABS_PATH)/DG_Pred_test.csv
DG_1_SAVE_FILE_NAME_TEST = $(ABS_PATH)/DG_1_Pred_test.csv


eval_val = val 
eval_test = test

device = cpu

all: run_qg_val run_qg_test run_qag_val run_qag_test run_dg_val run_dg_test run_dg_1_test run_dg_1_val

# QG Runner
run_qg_val:
	python save_predictions.py --model $(MODEL_QG_BASE) --model_type $(QG_TYPE) \
	--file_path $(QG_PATH) --save_file_name $(QG_SAVE_FILE_NAME_VAL) --eval_type $(eval_val) \
	--device $(device)

run_qg_test:
	python save_predictions.py --model $(MODEL_QG_BASE) --model_type $(QG_TYPE) \
	--file_path $(QG_PATH) --save_file_name $(QG_SAVE_FILE_NAME_TEST) --eval_type $(eval_test) \
	--device $(device)

# QAG Runner
run_qag_val:
	python save_predictions.py --model $(MODEL_QAG_BASE) --model_type $(QAG_TYPE) \
	--file_path $(QG_PATH) --save_file_name $(QAG_SAVE_FILE_NAME_VAL) --eval_type $(eval_val) \
	--device $(device)

run_qag_test:
	python save_predictions.py --model $(MODEL_QAG_BASE) --model_type $(QAG_TYPE) \
	--file_path $(QG_PATH) --save_file_name $(QAG_SAVE_FILE_NAME_TEST) --eval_type $(eval_test) \
	--device $(device)

# Distractor_runner
run_dg_val: 
	python save_predictions.py --model $(MODEL_DG_BASE) --model_type $(DG_TYPE) \
	--file_path $(DG_PATH) --save_file_name $(DG_SAVE_FILE_NAME_VAL) --eval_type $(eval_val) \
	--device $(device)

run_dg_test: 
	python save_predictions.py --model $(MODEL_DG_BASE) --model_type $(DG_TYPE) \
	--file_path $(DG_PATH) --save_file_name $(DG_SAVE_FILE_NAME_TEST) --eval_type $(eval_test) \
	--device $(device)

# Distractor_1_ runner
run_dg_1_val:
	python save_predictions.py --model $(MODEL_DG_1_SMALL) --model_type $(DG_1_TYPE) \
	--file_path $(DG_1_PATH) --save_file_name $(DG_1_SAVE_FILE_NAME_VAL) --eval_type $(eval_val) \
	--device $(device)

run_dg_1_test:
	python save_predictions.py --model $(MODEL_DG_1_SMALL) --model_type $(DG_1_TYPE) \
	--file_path $(DG_1_PATH) --save_file_name $(DG_1_SAVE_FILE_NAME_TEST) --eval_type $(eval_test) \
	--device $(device)
