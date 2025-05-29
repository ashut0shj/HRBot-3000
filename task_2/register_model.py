from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="4890ff6c-9baa-4f48-b125-fd0bf57adaa8",
    resource_group_name="HR-Analytics-RG",
    workspace_name="EmployeeSentimentAnalysis"
)

model = Model(
    name="best_attrition_model",
    path="azureml://jobs/cool_rabbit_rk8jm3jnfw/outputs/artifacts/paths/model/",  # your model path here
    type="mlflow_model"
)

registered_model = ml_client.models.create_or_update(model)

print(f"Model registered: {registered_model.name}, version: {registered_model.version}")
