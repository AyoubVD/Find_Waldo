from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model
from azureml.core.compute import AksCompute
from azureml.core.workspace import Workspace
from azureml.core.model import InferenceConfig
from os import getcwd as gc

# Choose the webservice you are interested in
# myservice
# aks-waldo
# wally
ws = Workspace.from_config()
service = Webservice(ws, 'myservice')
print(service.get_logs())