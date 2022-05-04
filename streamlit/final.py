import os
import json
import requests
import IPython
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import azureml.core
from azureml.core import Dataset, Run
from azureml.core import Experiment
from azureml.core.workspace import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice
from azureml.automl.core.shared.constants import ImageTask
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, choice
from azureml.train.hyperdrive import BanditPolicy, RandomParameterSampling
from azureml.train.hyperdrive import uniform
from azureml.data import DataType


k = os.getcwd()
print(k)

#------------------------
st.title("Let's find Wally 🔎")
uploaded_file = None
formats = ['.png', '.jpeg', '.jpg', '.bmp', '.raw', '.tiff']

uploaded_file = st.file_uploader("Choose an image",help = "Insert a picture which you want to check wether or not Wally is in it.", type = formats)

if(uploaded_file!=None):
    st.write("This is the image you've chosen")
    st.image(uploaded_file)
hom_image = "wally.png"
st.image(hom_image)
# cd "C:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo\"
#streamlit run streamlit/final.py
#------------------------

ws = Workspace.from_config()
cluster_name = "gpu-cluster-nc6"

try:
    compute_target = ws.compute_targets[cluster_name]
    print('Found existing compute target.')
except KeyError:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', 
                                                           idle_seconds_before_scaledown=1800,
                                                           min_nodes=0, 
                                                           max_nodes=4)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

#automl-wally-image-multiclass
experiment_name = 'final' 
experiment = Experiment(ws, name=experiment_name)
sample_image = "./streamlit/images/wally/w1.jpg"
IPython.display.Image(filename=sample_image)

src = "./streamlit/images/"
train_validation_ratio = 5

# Retrieving default datastore that got automatically created when we setup a workspace
workspaceblobstore = ws.get_default_datastore().name

# Path to the training and validation files
train_annotations_file = os.path.join(src, "train_annotations.jsonl")
validation_annotations_file = os.path.join(src, "validation_annotations.jsonl")

# sample json line dictionary
json_line_sample = {
    "image_url": "AmlDatastore://"
    + workspaceblobstore
    + "/"
    + os.path.basename(os.path.dirname(src)),
    "label": "",
}

index = 0
# Scan each sub directary and generate jsonl line
with open(train_annotations_file, "w") as train_f:
    with open(validation_annotations_file, "w") as validation_f:
        for className in os.listdir(src):
            subDir = src + className
            if not os.path.isdir(subDir):
                continue
            # Scan each sub directary
            print("Parsing " + subDir)
            for image in os.listdir(subDir):
                json_line = dict(json_line_sample)
                json_line["image_url"] += f"/{className}/{image}"
                json_line["label"] = className

                if index % train_validation_ratio == 0:
                    # validation annotation
                    validation_f.write(json.dumps(json_line) + "\n")
                else:
                    # train annotation
                    train_f.write(json.dumps(json_line) + "\n")
                index += 1
                
# Retrieving default datastore that got automatically created when we setup a workspace
ds = ws.get_default_datastore()
#we use this to upload all the files we need
#ds.upload(src_dir="./streamlit/images", target_path="images")

# get existing training dataset
training_dataset_name = "imagestraining"
if training_dataset_name in ws.datasets:
    training_dataset = ws.datasets.get(training_dataset_name)
    print("Found the training dataset", training_dataset_name)
else:
    # create training dataset
    training_dataset = Dataset.Tabular.from_json_lines_files(
        path=ds.path("images/train_annotations.jsonl"),
        set_column_types={"image_url": DataType.to_stream(ds.workspace)},
    )
    training_dataset = training_dataset.register(
        workspace=ws, name=training_dataset_name
    )
# get existing validation dataset
validation_dataset_name = "imagesvalidation"
if validation_dataset_name in ws.datasets:
    validation_dataset = ws.datasets.get(validation_dataset_name)
    print("Found the validation dataset", validation_dataset_name)
else:
    # create validation dataset
    validation_dataset = Dataset.Tabular.from_json_lines_files(
        path=ds.path("images/validation_annotations.jsonl"),
        set_column_types={"image_url": DataType.to_stream(ds.workspace)},
    )
    validation_dataset = validation_dataset.register(
        workspace=ws, name=validation_dataset_name
    )
print("Training dataset name: " + training_dataset.name)
print("Validation dataset name: " + validation_dataset.name)

training_dataset.to_pandas_dataframe()


image_config_vit = AutoMLImageConfig(
    task=ImageTask.IMAGE_CLASSIFICATION,
    compute_target=compute_target,
    training_data=training_dataset,
    validation_data=validation_dataset,
    hyperparameter_sampling=GridParameterSampling({"model_name": choice("vitb16r224")}),
    iterations=1,
)

automl_image_run = experiment.submit(image_config_vit)
automl_image_run.wait_for_completion(wait_post_processing=True)


parameter_space = {
    "learning_rate": uniform(0.001, 0.01),
    "model": choice(
        {
            "model_name": choice("vitb16r224", "vits16r224"),
            "number_of_epochs": choice(15, 30),
        },
        {
            "model_name": choice("seresnext", "resnest50"),
            "layers_to_freeze": choice(0, 2),
        },
    ),
}

tuning_settings = {
    "iterations": 10,
    "max_concurrent_iterations": 2,
    "hyperparameter_sampling": RandomParameterSampling(parameter_space),
    "early_termination_policy": BanditPolicy(
        evaluation_interval=2, slack_factor=0.2, delay_evaluation=6
    ),
}

automl_image_config = AutoMLImageConfig(
    task=ImageTask.IMAGE_CLASSIFICATION,
    compute_target=compute_target,
    training_data=training_dataset,
    validation_data=validation_dataset,
    **tuning_settings,
)

automl_image_run = experiment.submit(automl_image_config)
automl_image_run.wait_for_completion(wait_post_processing=True)

hyperdrive_run = Run(experiment=experiment, run_id=automl_image_run.id + "_HD")
hyperdrive_run

best_child_run = hyperdrive_run.get_best_child()
model_name = best_child_run.properties["model_name"]
model = best_child_run.register_model(
    model_name=model_name, model_path="outputs/model.pt"
)

# Choose a name for your cluster
aks_name = "aks-cpu-mc"
# Check to see if the cluster already exists
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print("Found existing compute target")
except ComputeTargetException:
    print("Creating a new compute target...")
    # Provision AKS cluster with a CPU machine
    prov_config = AksCompute.provisioning_configuration(vm_size="STANDARD_D3_V2")
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )
    aks_target.wait_for_completion(show_output=True)
    

best_child_run.download_file(
    "outputs/scoring_file_v_1_0_0.py", output_file_path="score.py"
)
environment = best_child_run.get_environment()
inference_config = InferenceConfig(entry_script="score.py", environment=environment)

aks_config = AksWebservice.deploy_configuration(
    autoscale_enabled=True, cpu_cores=1, memory_gb=5, enable_app_insights=True
)

aks_service = Model.deploy(
    ws,
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=aks_target,
    name="automl-image-test-cpu-mc",
    overwrite=True,
)

aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
resp = kowalski(aks_service, "test7.png")

def kowalski(aks_service, sample_image):

    # URL for the web service
    scoring_uri = aks_service.scoring_uri

    # If the service is authenticated, set the key or token
    key, _ = aks_service.get_keys()
    #sample_image = "test7.png"

    # Load image data
    data = open(sample_image, "rb").read()

    # Set the content type
    headers = {"Content-Type": "application/octet-stream"}

    # If authentication is enabled, set the authorization header
    headers["Authorization"] = f"Bearer {key}"

    # Make the request and display the response
    resp = requests.post(scoring_uri, data, headers=headers)
    print(resp.text)
    IMAGE_SIZE = (18, 12)
    plt.figure(figsize=IMAGE_SIZE)
    img_np = mpimg.imread(sample_image)
    img = Image.fromarray(img_np.astype("uint8"), "RGB")
    x, y = img.size

    fig, ax = plt.subplots(1, figsize=(15, 15))
    # Display the image
    ax.imshow(img_np)

    prediction = json.loads(resp.text)
    label_index = np.argmax(prediction["probs"])
    label = prediction["labels"][label_index]
    conf_score = prediction["probs"][label_index]

    display_text = "{} ({})".format(label, round(conf_score, 3))
    print(display_text)

    color = "red"
    plt.text(30, 30, display_text, color=color, fontsize=30)
    plt.savefig('./streamlit/output.png')
    plt.show()
    return resp

print('Please upload a file using the web app (づ ◕‿◕ )づ')

if(uploaded_file!=None):
    resp = kowalski(aks_service, uploaded_file)
    st.write(resp)
    st.image('./streamlit/output.png')

