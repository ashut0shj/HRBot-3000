# setup_azure_ml.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    Environment, 
    Model, 
    ManagedOnlineEndpoint, 
    ManagedOnlineDeployment, 
    CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes
import os
import time
from pathlib import Path

# Configuration
SUBSCRIPTION_ID = "4890ff6c-9baa-4f48-b125-fd0bf57adaa8"
RESOURCE_GROUP = "hr-sentiment-rg"
WORKSPACE_NAME = "hr-sentiment-ws"
ENDPOINT_NAME = "attrition-endpoint"
MODEL_NAME = "attrition-model"
ENV_NAME = "attrition-env"
DEPLOYMENT_NAME = "attrition-deployment"
LOCATION = "eastus2"

def verify_model_files():
    """Verify all required model files exist"""
    required_files = [
        "model/attrition_model.pkl",
        "model/performance_encoder.pkl",
        "model/scaler.pkl",
        "model/app.py",
        "model/suggestions.py",  # Fixed typo from 'suggestions.py'
        "model/conda.yml",  # Fixed typo from 'conda.yml'
        "model/.env"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

def deploy_model():
    """Main deployment function"""
    print("Starting Azure ML deployment...")
    
    # Verify files first
    print("Verifying model files...")
    verify_model_files()
    
    # Initialize client
    print("Initializing ML Client...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    # Create environment
    print("Creating environment...")
    env = Environment(
        name=ENV_NAME,
        description="Environment for employee attrition prediction",
        conda_file="model/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    ml_client.environments.create_or_update(env)
    
    # Register model
    print("Registering model...")
    model = Model(
        path="model",
        name=MODEL_NAME,
        description="Employee attrition prediction model",
        type=AssetTypes.CUSTOM_MODEL
    )
    ml_client.models.create_or_update(model)
    
    # Create endpoint (using ManagedOnlineEndpoint instead of OnlineEndpoint)
    print("Creating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Endpoint for employee attrition prediction",
        auth_mode="key",
        location=LOCATION
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Create deployment (using ManagedOnlineDeployment)
    print("Creating deployment...")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=MODEL_NAME,
        environment=f"{ENV_NAME}@latest",
        code_configuration=CodeConfiguration(
            code="model",
            scoring_script="app.py"
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1
    )
    
    print("Starting deployment (this may take 15-20 minutes)...")
    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    
    # Wait for completion with progress updates
    while not poller.done():
        time.sleep(30)
        print("Deployment in progress...")
    
    poller.result()
    print("Deployment completed successfully!")
    
    # Set traffic
    print("Setting traffic...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Get endpoint details
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys = ml_client.online_endpoints.list_keys(ENDPOINT_NAME)
    
    print("\n=== DEPLOYMENT SUCCESS ===")
    print(f"Endpoint name: {endpoint.name}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"Primary key: {keys.primary_key}")
    print("=========================")

if __name__ == "__main__":
    try:
        deploy_model()
    except Exception as e:
        print(f"\nDeployment failed: {str(e)}")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Verify all model files exist in the 'model' directory")
        print("2. Check your Azure permissions and quota")
        print("3. Ensure the scoring script (app.py) is in the model folder")
        print("4. Try a different instance type if deployment fails")
        print("5. Check Azure portal for detailed error logs")
        print("6. Make sure you're using the latest Azure ML SDK: pip install --upgrade azure-ai-ml")