from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model
from azureml.core.compute import AksCompute
from azureml.core.workspace import Workspace
from azureml.core.model import InferenceConfig
from os import getcwd as gc

ws = Workspace.from_config()
aks_target = AksCompute(ws,"tizniay2")
# If deploying to a cluster configured for dev/test, ensure that it was created with enough
# cores and memory to handle this deployment configuration. Note that memory is also used by
# things such as dependencies and AML components.
#deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
model = Model(ws, 'AutoMLfbf830ae82', version=2)
print(gc())

service = Model.serialize("http://b333892e-92cc-4e82-9844-ff522127be62.westeurope.azurecontainer.io/score")#(ws, "myservice", [model],inference_config = InferenceConfig(entry_script='./Stage/Arinti/FindWaldo/FindWaldo/streamlit/score.py'))

service.wait_for_deployment(show_output = True)
print(service.state)
print(service.get_logs())