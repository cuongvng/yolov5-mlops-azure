## Build CI/CD pipelines for YOLOv5 object detection service on Azure.

### Repo Structure
- Generated from [this template](https://github.com/microsoft/MLOpsPython).
- `.pipelines`: contains YAML files describing the CI/CD pipeline on Azure DevOps.
- `bootstrap`: scripts used to bootstrap a new project from the template.
- `docs`: original how-to guide to setup a CI/CD on Azure, with a simple regression model.
- `environment_setup`: YAML files used to setup cloud enviroment, provision resources, etc.
- `ml_service`: scripts using Azure Python SDK to build training, registration and evaluation pipelines on Azure Machine Learning Studio.
- `yolov5`: main scripts to interact with Azure Virtual Machines, containing the original [YOLOv5 code](https://github.com/ultralytics/yolov5).
