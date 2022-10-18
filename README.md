## Build CI/CD pipelines for YOLOv5 object detection service on Azure.

### Repo Structure
- Generated from [this template](https://github.com/microsoft/MLOpsPython).
- `.pipelines`: contains YAML files describing the CI/CD pipeline on Azure DevOps.
- `bootstrap`: scripts used to bootstrap a new project from the template.
- `docs`: original how-to guide to setup a CI/CD on Azure, with a simple regression model.
- `environment_setup`: YAML files used to setup cloud enviroment, provision resources, etc.
- `ml_service`: scripts using Azure Python SDK to build training, registration and evaluation pipelines on Azure Machine Learning Studio.
- `yolov5`: main scripts to interact with Azure Virtual Machines, containing the original [YOLOv5 code](https://github.com/ultralytics/yolov5).

### Setup CI pipeline:
- *Basically, I created an Azure DevOps pipeline that was triggered whenever a commit is pushed/merged to the `master` branch, to run another pipeline on Azure Machine Learning Studio that would retrain my yolov5 model*.
- Configure the `.pipeline/yolov5-ci.yml`, which trigger the build pipeline for model training. - Write training code
- Setup `.env`
- Register `coco128` dataset on Datastores, which would be used to quickly train the model (since this repo's purpose is to demonstrate how to setup a CI/CD pipeline, I will not waste time on training the model, just use a small dataset).
- Config training params in `parameters.json`
- Write eval code: evaluate the newly trained mode, if its metric is better than the current model on production, register that model to Azure ML Studio, otherwise cancel the pipeline run!
- Eval metric: mean Average Precision under IoU 0.5 to 0.95 `mAP_0.5_0.95`, which will be saved in a tag during the training process, just read that tag to get the current model’s metric.

<!-- <details>
<summary>Click to expand/collapse guides</summary>
  

- 

</details> -->

### Setup CD pipeline
- Create **a new Azure DevOps pipeline**  specified by `yolov5_cd.yml`, which would be triggered whenever the CI pipeline is done.
- Automatically deploy the newly registered model on Azure ML Realtime Endpoint, both managed compute instance and Kubernetes Services.
