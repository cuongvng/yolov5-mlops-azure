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
from operator import mod
import os
import json
from pyexpat import model
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azure.ai.ml import MLClient
import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# from azure.identity import DefaultAzureCredential

import subprocess
# Install cv2 deps (missing in the Docker container)
subprocess.run(["apt-get", "update"])
subprocess.run(["apt-get", "install", "ffmpeg", "libsm6", "libxext6",  "-y"])

import sys
FILE = Path(__file__).resolve()
YOLOV5_PATH = os.path.join(str(FILE.parents[0]), "yolov5_repo")  # YOLOv5 dependency
print(("YOLOV5_PATH", YOLOV5_PATH))
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH)) 

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import LOGGER, Profile, check_img_size, cv2, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.dataloaders import LoadImages


# from inference_schema.schema_decorators \
#     import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type \
#     import NumpyParameterType

# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json

subprocess.run(["pip", "install", "-r", os.path.join(YOLOV5_PATH, "requirements.txt")])

def init():
    global model

    device = select_device("")
    model_name = os.environ.get("MODEL_NAME")
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    sp_id = os.environ.get("SP_APP_ID")
    sp_secret = os.environ.get("SP_APP_SECRET")

    svc_pr = ServicePrincipalAuthentication(
       tenant_id=tenant_id,
       service_principal_id=sp_id,
       service_principal_password=sp_secret)

    ws = Workspace(
        workspace_name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=svc_pr
    )

    model_path = Model.get_model_path(
        model_name=model_name, 
        version=None, 
        _workspace=ws
    )
    
    print("model_path", model_path)
    model = DetectMultiBackend(
        weights=model_path, 
        device=device, 
        data=os.path.join(YOLOV5_PATH, "data/coco128.yaml")
        )

def run(input, request_headers):
    img_link = json.loads(input)["image_link"]
    print("img_link", img_link)

    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    summary = ""

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size([320, 320], s=stride)  # check image size
    bs = 1 # batch_size
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    if not os.path.exists("./outputs"):
        save_dir = os.mkdir("./outputs")
    else:
        save_dir = "./outputs"

    # Download image from link and create dataset from it
    file_name = img_link.split('/')[-1]
    source = os.path.join(YOLOV5_PATH, file_name)
    print(("Source", source))
    subprocess.run(["curl", img_link, "--output", source])
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = path, im0s.copy()

            p = Path(p)  # to Path
            save_path = str(os.path.join(save_dir, file_name)) 
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Save results (image with detections)
            cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        summary += f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms\n" 
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms\n")

    subprocess.run(["rm", source])
    # # Demonstrate how we can log custom data into the Application Insights
    # # traces collection.
    # # The 'X-Ms-Request-id' value is generated internally and can be used to
    # # correlate a log entry with the Application Insights requests collection.
    # # The HTTP 'traceparent' header may be set by the caller to implement
    # # distributed tracing (per the W3C Trace Context proposed specification)
    # # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               summary
    ))

    return {"result": summary}


if __name__ == "__main__":
    input = "{'image_link': 'https://raw.githubusercontent.com/cuongvng/yolov5/master/data/images/zidane.jpg'}"
    run(json.loads(input), {})
