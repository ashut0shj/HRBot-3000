
from azure.ai.ml import MLClient, automl
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes


ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="4890ff6c-9baa-4f48-b125-fd0bf57adaa8",
    resource_group_name="HR-Analytics-RG",
    workspace_name="EmployeeSentimentAnalysis"
)



automl_job = automl.classification(
    experiment_name="automl_attrition",
    training_data=Input(
        type=AssetTypes.URI_FILE,
        path=ml_client.data.get(name="employee_attrition_data", version="1").path
    ),
    target_column_name="EmployeeStatus",
    primary_metric="AUC_weighted",
    compute="cpu-cluster-small",
    enable_model_explainability=True
)

job = ml_client.jobs.create_or_update(automl_job)

print(f"Submitted AutoML job with ID: {job.name}")
print(f"Job status: {job.status}")