# DeepFace Model Deployment Scripts

## Folder Architecture

```
DeepFace/
├── DeployScript/
│   ├── deploy.py              # Deploy DeepFace model to Azure ML
│   ├── delete_deployment.py   # Delete deployment and endpoint
│   └── README.md              # Documentation
├── Environment/
│   └── conda.yaml             # Environment dependencies
├── ScoringScript/
│   └── score.py               # Scoring script for inference
```

## How to Run deploy.py and delete_deployment.py

### Deploy Model

```bash
python scripts\host_model\DeepFace\DeployScript\deploy.py `
  --env-folder scripts\host_model\DeepFace\Environment `
  --scoring-folder scripts\host_model\DeepFace\ScoringScript `
  --model-weights temporary\cache\.deepface\weights
```

**Options:**
- `--env-folder` - Path to environment folder with `conda.yaml`
- `--scoring-folder` - Path to scoring script folder with `score.py`
- `--model-weights` - (Optional) Path to model weights directory. If not provided, models are auto-downloaded from GitHub
- `--endpoint-name` - (Optional) Custom endpoint name (default: `deepface-endpoint`)
- `--instance-type` - (Optional) Azure compute instance type (default: `Standard_F4s_v2`)
- `--instance-count` - (Optional) Number of instances (default: `1`)

### Delete Deployment

```bash
python scripts\host_model\DeepFace\DeployScript\delete_deployment.py --force
```

**Options:**
- `--endpoint-name` - (Optional) Endpoint name to delete (default: `deepface-endpoint`)
- `--deployment-name` - (Optional) Deployment name to delete (default: `blue`)
- `--force` - Skip confirmation prompt
- `--keep-endpoint` - Delete only deployment, keep endpoint

### Prerequisites

Set up Azure credentials in `.env` file:
```env
AZURE_TENANT_ID=<value>
AZURE_CLIENT_ID=<value>
AZURE_CLIENT_SECRET=<value>
AZURE_SUBSCRIPTION_ID=<value>
AZURE_RESOURCE_GROUP=<value>
AZUREML_WORKSPACE_NAME=<value>
```
