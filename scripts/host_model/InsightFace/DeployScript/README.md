# InsightFace Model Deployment Scripts

## Overview

This folder contains deployment scripts for deploying InsightFace models to Azure Machine Learning Online Endpoints. The scripts are fully idempotent and will only create assets if they don't already exist.

## Folder Architecture

```
InsightFace/
├── DeployScript/
│   ├── deploy.py              # Deploy InsightFace model to Azure ML
│   ├── delete_deployment.py   # Delete deployment and endpoint
│   └── README.md              # Documentation
├── Environment/
│   └── conda.yaml             # Environment dependencies
├── ScoringScript/
│   └── score.py               # Scoring script for inference
```

## Prerequisites

### 1. Install Required Packages

```bash
pip install azure-ai-ml azure-identity python-dotenv
```

### 2. Set Up Azure Credentials

Create a `.env` file in your project root with the following environment variables:

```env
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-client-id>
AZURE_CLIENT_SECRET=<your-client-secret>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZUREML_WORKSPACE_NAME=<your-workspace-name>
```

### 3. Prepare Environment and Scoring Script

Ensure you have:
- **Environment folder** with `conda.yaml` - Specifies Python dependencies for InsightFace
- **Scoring Script folder** with `score.py` - Contains the inference logic

## How to Use

### Deploy Model

```bash
python scripts\host_model\InsightFace\DeployScript\deploy.py `
  --env-folder scripts\host_model\InsightFace\Environment `
  --scoring-folder scripts\host_model\InsightFace\ScoringScript
```

**Optional Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env-folder` | Path to folder with `conda.yaml` | **Required** |
| `--scoring-folder` | Path to folder with `score.py` | **Required** |
| `--model-path` | Path to model directory (optional) | HuggingFace/cached |
| `--endpoint-name` | Custom endpoint name | `insightface-endpoint` |
| `--model-name` | Model registry name | `insightface-model` |
| `--model-version` | Model version | `1` |
| `--env-name` | Environment name | `insightface-online-endpoint-environment` |
| `--deployment-name` | Deployment name | `blue` |
| `--instance-type` | Azure compute instance type | `Standard_F4s_v2` |
| `--instance-count` | Number of instances | `1` |
| `--no-save-credentials` | Don't save credentials to .env | (flag) |

**Examples:**

```bash
# Basic deployment with defaults
python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript

# With custom endpoint name
python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript `
                 --endpoint-name my-insightface-endpoint

# Custom instance configuration
python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript `
                 --instance-type Standard_D4s_v3 --instance-count 2

# With custom model directory
python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript `
                 --model-path ./models/insightface_buffalo_l
```

### Delete Deployment

```bash
python scripts\host_model\InsightFace\DeployScript\delete_deployment.py --force
```

**Optional Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--endpoint-name` | Endpoint name to delete (default: `insightface-endpoint`) |
| `--deployment-name` | Deployment name to delete (default: `blue`) |
| `--force` | Skip confirmation prompt |
| `--keep-endpoint` | Delete only deployment, keep endpoint |

**Safety Features:**
- Requires explicit confirmation (type 'DELETE' or use `--force` flag)
- Alternatively, set `CONFIRM_DELETE=true` environment variable
- Routes 0% traffic to deployment before deletion
- Gracefully handles missing resources

**Examples:**

```bash
# Interactive mode (will ask for confirmation)
python delete_deployment.py

# Force delete without confirmation
python delete_deployment.py --force

# Using environment variable
$env:CONFIRM_DELETE='true'
python delete_deployment.py

# Delete only deployment, keep endpoint
python delete_deployment.py --keep-endpoint --force
```

## InsightFace Model Information

The default configuration deploys the **buffalo_l** model variant, which includes:
- Face detection (RetinaFace)
- Face recognition (ArcFace)
- Facial landmark detection
- Age and gender estimation

Models are typically downloaded from HuggingFace at runtime or cached locally, unless a custom `--model-path` is provided.

## What the Scripts Do

### deploy.py

1. **Validates paths** - Ensures conda.yaml and score.py exist
2. **Creates/gets model asset** - Registers model in Azure ML registry
3. **Creates/gets environment** - Sets up conda environment with dependencies
4. **Creates/gets endpoint** - Provisions managed online endpoint
5. **Creates/gets deployment** - Deploys model to endpoint with code
6. **Routes traffic** - Directs 100% traffic to deployment
7. **Saves credentials** - Stores endpoint URI and API key to .env file

All steps are **idempotent** - they check if assets exist first and only create them if missing.

### delete_deployment.py

1. **Asks for confirmation** - Safety feature to prevent accidental deletion
2. **Routes traffic to 0%** - Stops routing requests to the deployment
3. **Deletes deployment** - Removes the deployment from the endpoint
4. **Deletes endpoint** - Optionally removes the entire endpoint
5. **Logs all actions** - Provides clear status messages

## Output

After successful deployment, credentials are saved to `.env`:

```env
INSIGHTFACE_ENDPOINT_URI=https://your-endpoint.eastus.inference.ml.azure.com/score
INSIGHTFACE_ENDPOINT_KEY=your-api-key-here
```

Use these values to make inference requests to your deployed model.

## Troubleshooting

### Missing Environment Variables

**Error:** `Missing required environment variable: AZURE_TENANT_ID`

**Solution:** Ensure all required variables are set in `.env` file and the file is in the directory where you run the script.

### Path Not Found

**Error:** `Environment folder not found`

**Solution:** Use absolute or relative paths correctly:
```bash
# Correct relative path usage
python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript

# Or use absolute paths
python deploy.py --env-folder C:\path\to\Environment --scoring-folder C:\path\to\ScoringScript
```

### Deployment Timeout

If deployment takes too long, you can check the status in Azure ML Studio dashboard. The default timeout is 60 seconds for requests.

### Permission Denied

**Error:** `Authentication error`

**Solution:** Verify that your Azure credentials have appropriate permissions in the resource group and workspace.

## Cost Considerations

- Managed endpoints incur costs on compute instances used
- Use `delete_deployment.py` to remove endpoints when not needed
- Monitor usage in Azure Portal
- Consider using `--keep-endpoint` flag to preserve endpoint while deleting deployment
