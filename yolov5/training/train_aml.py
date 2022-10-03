"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from azureml.core.run import Run
from azureml.core import Dataset
import os
import argparse
import yaml 
import pandas as pd
import json

import subprocess
print("print working dir:")
subprocess.run(["pwd"])
subprocess.run(["ls"])

YOLOV5_PATH = "./yolov5_repo"

def main():
	print("Running train_aml.py")

	parser = argparse.ArgumentParser("train")
	parser.add_argument(
		"--model_name",
		type=str,
		help="Name of the Model",
		default="yolov5_model.pt",
	)

	parser.add_argument(
		"--step_output",
		type=str,
		help=("output for passing data to next step")
	)

	parser.add_argument(
		"--dataset_version",
		type=str,
		help=("dataset version")
	)

	parser.add_argument(
		"--caller_run_id",
		type=str,
		help=("caller run id, for example ADF pipeline run id")
	)

	parser.add_argument(
		"--dataset_name",
		type=str,
		help=("Dataset name. Dataset must be passed by name\
			  to always get the desired dataset version\
			  rather than the one used while the pipeline creation")
	)

	args = parser.parse_args()

	print("Argument [model_name]: %s" % args.model_name)
	print("Argument [step_output]: %s" % args.step_output)
	print("Argument [dataset_version]: %s" % args.dataset_version)
	print("Argument [caller_run_id]: %s" % args.caller_run_id)
	print("Argument [dataset_name]: %s" % args.dataset_name)

	model_name = args.model_name
	step_output_path = args.step_output
	dataset_version = args.dataset_version
	dataset_name = args.dataset_name

	run = Run.get_context()

	print("Getting training parameters")

	# Load the training parameters from the parameters file
	with open("parameters.json") as f:
		pars = json.load(f)
	try:
		train_args = pars["training"]
	except KeyError:
		print("Could not load training values from file")
		train_args = {}

	# Log the training parameters
	print(f"Parameters: {train_args}")
	for (k, v) in train_args.items():
		run.log(k, v)
		run.parent.log(k, v)

	# Get the dataset
	if (dataset_name):
		dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
	else:
		e = ("No dataset provided")
		print(e)
		raise Exception(e)

	# Link dataset to the step run so it is trackable in the UI
	run.input_datasets['training_data'] = dataset
	run.parent.tag("dataset_id", value=dataset.id)

	data_description_file = os.path.join(YOLOV5_PATH, 'data/coco128.yaml')
	with open(data_description_file, 'r') as f:
		cfg = yaml.safe_load(f)
		dest = cfg["path"]
	dataset.download(target_path=dest)

	# Call training command from the original yolov5 repo, e.g.
	# $ python ./yolov5_repo/train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt

	import subprocess
	subprocess.run(["python", os.path.join(YOLOV5_PATH, "train.py"), 
				"--img", f"{train_args['img_size']}",
				"--batch", f"{train_args['batch_size']}",
				"--epochs", f"{train_args['n_epochs']}",
				"--data", f"{data_description_file}",
				"--weights", f"{train_args['weights']}"])

	# Load saved model and metrics 
	training_res_path = os.join(YOLOV5_PATH, "runs/training/exp/")
	model_path = os.join(training_res_path, "weights/best.pt")
	metric_path = os.join(training_res_path, "results.csv")

	# Copy the model to a new dir `step_output_path`, and delete the `runs/training/exp/` dir, so retraining won't generate a new dir (`exp2`, `exp3`, etc.)

	os.makedirs(step_output_path, exist_ok=True)
	model_output_path = os.path.join(step_output_path, model_name)
	subprocess.run(["cp", model_path, model_output_path])

	# Also copy it to the special `outputs` dir in the Azure VM, all content in this directory is automatically uploaded to the ML workspace.
	output_path = os.path.join('outputs', model_name)
	subprocess.run(["cp", model_path, output_path])
	subprocess.run(["rm", "-rf", training_res_path])

	# Log the metrics returned from the train function
	metrics = pd.read_csv(metric_path)
	name = "mAP_0.5:0.95"
	mAP_50_95 = metrics.iloc[-1]["metrics/"+name]
	run.log(name, mAP_50_95)
	run.parent.log(name, mAP_50_95)

	run.tag("run_type", value="train")
	print(f"tags now present for run: {run.tags}")

	run.complete()


if __name__ == '__main__':
	main()
