# setup_azure_ml.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Initialize ML Client
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="4890ff6c-9baa-4f48-b125-fd0bf57adaa8",
    resource_group_name="try",
    workspace_name="hr-sentiment-ws"
)

# Create workspace if it doesn't exist
from azure.ai.ml.entities import Workspace
ws = Workspace(
    name="hr-sentiment-ws",
    location="eastus2",
    resource_group="hr-sentiment-rg"
)
ml_client.workspaces.begin_create_or_update(ws)