from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Connect to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="4890ff6c-9baa-4f48-b125-fd0bf57adaa8",
    resource_group_name="HR-Analytics-RG",
    workspace_name="EmployeeSentimentAnalysis",
)

# Upload datasets
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Define dataset
hr_dataset = Data(
    path="merged_employee_data.csv",
    type=AssetTypes.URI_FILE,
    name="employee_attrition_data",
    description="HR dataset for attrition prediction",
)

# Upload
ml_client.data.create_or_update(hr_dataset)