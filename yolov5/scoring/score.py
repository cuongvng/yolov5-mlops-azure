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
from azureml.core.model import Model
import sys
sys.path.apped("../yolov5_repo")
from detect import main as yolo_main
import subprocess

# from inference_schema.schema_decorators \
#     import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type \
#     import NumpyParameterType

# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json

YOLOV5_PATH = "./yolov5_repo"
file_name = "zidane.jpg"
subprocess.run(["pip", "install", "-r", YOLOV5_PATH + "/requirements.txt"])
test_img = os.path.join("./scoring", file_name)

def run(input, request_headers):
    img_link = input["image_link"]
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])
    print("model_path", model_path)

    prj_path = os.path.join(YOLOV5_PATH, "runs/detect")
    
    # subprocess.run(["python", 
    #                 os.path.join(YOLOV5_PATH, "detect.py"),
    #                 "--weights", model_path,
    #                 "--source", img_link,
    #                 "--project", prj_path,
    #                 "--img", 320
    #                 ])

    opt = {
        "weights": model_path,
        "source": img_link,
        "project": prj_path,
        "img": 320
    }
    summary = yolo_main(opt)

    # Bounded image is saved to `res_path/exp/file_name`
    output_file = os.path.join(prj_path, "exp", file_name)
    print("output_file", output_file)
    subprocess.run(["cp", output_file, os.path.join("outputs", file_name)])

    # Delete the output dir to avoid increment paths in later runs
    subprocess.run(["rm", "-rf", os.path.join(prj_path, "exp")])


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
    input = "{'img_link': 'https://raw.githubusercontent.com/cuongvng/yolov5/master/data/images/zidane.jpg'}"
    run(json.loads(input), {})
