import subprocess
import datetime
import argparse

classification_tasks = [
    "VH immune2vec_H_FULL_25","VH immune2vec_H_FULL_50", "VH immune2vec_H_FULL_100", "VH immune2vec_H_FULL_150", "VH immune2vec_H_FULL_200" ,"VH immune2vec_H_FULL_500" ,"VH immune2vec_H_FULL_1000",
    "JH immune2vec_H_FULL_25", "JH immune2vec_H_FULL_50", "JH immune2vec_H_FULL_100", "JH immune2vec_H_FULL_150", "JH immune2vec_H_FULL_200" ,"JH immune2vec_H_FULL_500", "JH immune2vec_H_FULL_1000",
    "VL immune2vec_L_FULL_25", "VL immune2vec_L_FULL_50", "VL immune2vec_L_FULL_100", "VL immune2vec_L_FULL_150", "VL immune2vec_L_FULL_200" ,"VL immune2vec_L_FULL_500", "VL immune2vec_L_FULL_1000",
    "JL immune2vec_L_FULL_25", "JL immune2vec_L_FULL_50", "JL immune2vec_L_FULL_100", "JL immune2vec_L_FULL_150", "JL immune2vec_L_FULL_200", "JL immune2vec_L_FULL_500", "JL immune2vec_L_FULL_1000",
    "isoH immune2vec_H_FULL_25", "isoH immune2vec_H_FULL_50", "isoH immune2vec_H_FULL_100", "isoH immune2vec_H_FULL_150", "isoH immune2vec_H_FULL_200", "isoH immune2vec_H_FULL_500", "isoH immune2vec_H_FULL_1000" ,
    "isoL immune2vec_L_FULL_25", "isoL immune2vec_L_FULL_50", "isoL immune2vec_L_FULL_100", "isoL immune2vec_L_FULL_150", "isoL immune2vec_L_FULL_200", "isoL immune2vec_L_FULL_500" ,"isoL immune2vec_L_FULL_1000",
    "VH physicochemical", "JH physicochemical", "VL physicochemical" ,"JL physicochemical" , "isoH physicochemical" ,"isoL physicochemical",
    "VH frequency" ,"JH frequency", "VL frequency", "JL frequency", "isoH frequency", "isoL frequency",
    "VH esm2","JH esm2", "VL esm2", "JL esm2", "isoH esm2", "isoL esm2" ,
    "VH ProtT5", "JH ProtT5", "VL ProtT5", "JL ProtT5", "isoH ProtT5", "isoL ProtT5",
    "VH esm2_3B", "JH esm2_3B", "VL esm2_3B", "JL esm2_3B", "isoH esm2_3B", "isoL esm2_3B",
    "VH antiBERTy", "JH antiBERTy", "VL antiBERTy", "JL antiBERTy", "isoH antiBERTy", "isoL antiBERTy"
]

task_id = 73  # You can set the desired task ID or obtain it programmatically

print(f"[{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}] [start] Task ID: {task_id}")

task = classification_tasks[task_id - 1]
parser = argparse.ArgumentParser(description="Gene usage tasks")
parser.add_argument("gene", default= "VH antiBERTy" "JH antiBERTy" "VL antiBERTy" "JL antiBERTy" "isoH antiBERTy" "isoL antiBERTy",type=str, help="Gene type (VH, VL, JH, JL, isoH, isoL)")
parser.add_argument("embedding", default="antiBERTy",type=str, help="Type of embedding (immune2vec, esm2, ProtT5)")
parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=True)


# Run the Python script with the specified task
subprocess.run(["python", "8_run_classification_task.py", task])

print(f"[{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}] [end] Task ID: {task_id}")