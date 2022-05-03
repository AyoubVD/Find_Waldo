#connecting to the workspace
from azureml.core import Workspace
ws = Workspace(subscription_id="a8260178-3b6d-4bce-a07e-3aae8c7a62af",
               resource_group="RG-Internship-Ayoub",
               workspace_name="Wally")
# az account set -s "a8260178-3b6d-4bce-a07e-3aae8c7a62af"
# $GROUP="RG-Internship-Ayoub"
# $LOCATION="West Europe"
# "westeurope"
# $WORKSPACE="Wally"

# Deploy to ACI
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service = Model.deploy(ws, "aciservice", [model], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)
print(service.state)

# The recommended deployment pattern 
# is to create a deployment configuration object with the deploy_configuration method and 
# then use it with the deploy method of the Model class as shown below.
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


service_name = 'my-custom-env-service'

inference_config = InferenceConfig(entry_script='score.py', environment=environment)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                          name=service_name,
                          models=[model],
                          inference_config=inference_config,
                          deployment_config=aci_config,
                          overwrite=True)
service.wait_for_deployment(show_output=True)