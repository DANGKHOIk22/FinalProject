"""
Deploy InsightFace Model to Azure Machine Learning Online Endpoint

This script deploys (or updates) a Managed Online Endpoint in Azure ML for InsightFace scoring.
It is idempotent: it checks whether assets already exist (model/environment/endpoint/deployment)
and only creates them when missing.

Prerequisites:
    - Install: azure-ai-ml, azure-identity, python-dotenv
    - Set environment variables in .env file:
        - AZURE_TENANT_ID
        - AZURE_CLIENT_ID
        - AZURE_CLIENT_SECRET
        - AZURE_SUBSCRIPTION_ID
        - AZURE_RESOURCE_GROUP
        - AZUREML_WORKSPACE_NAME

Usage:
    python deploy.py --env-folder /path/to/env --scoring-folder /path/to/scoring
    
    Or with custom endpoint/model names:
    python deploy.py --env-folder /path/to/env --scoring-folder /path/to/scoring \\
                     --endpoint-name my-endpoint --model-name my-model

    Or with custom instance configuration:
    python deploy.py --env-folder /path/to/env --scoring-folder /path/to/scoring \\
                     --instance-type Standard_D4s_v3 --instance-count 2
"""

import os
import sys
import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv, set_key

# Azure ML SDK imports
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
    OnlineRequestSettings,
    Model,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.core.polling import LROPoller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from Azure SDK and urllib3
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class AzureMLInsightFaceDeployment:
    """Handles deployment of InsightFace model to Azure ML."""
    
    # Default configuration
    DEFAULT_ENDPOINT_NAME = "insightface-endpoint"
    DEFAULT_MODEL_NAME = "insightface-model"
    DEFAULT_MODEL_VERSION = "1"
    DEFAULT_ENV_NAME = "insightface-online-endpoint-environment"
    DEFAULT_ENV_VERSION = "2"
    DEFAULT_DEPLOYMENT_NAME = "blue"
    DEFAULT_INSTANCE_TYPE = "Standard_F4s_v2"
    DEFAULT_INSTANCE_COUNT = 1
    
    def __init__(
        self,
        env_folder: str,
        scoring_folder: str,
        endpoint_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        env_name: Optional[str] = None,
        env_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize InsightFace deployment handler.
        
        Args:
            env_folder: Path to the environment folder containing conda.yaml
            scoring_folder: Path to the scoring script folder
            endpoint_name: Azure ML endpoint name (default: insightface-endpoint)
            model_name: Model registry name (default: insightface-model)
            model_version: Model version (default: 1)
            env_name: Environment name (default: insightface-online-endpoint-environment)
            env_version: Environment version (default: 1)
            deployment_name: Deployment name (default: blue)
            instance_type: Compute instance type (default: Standard_F4s_v2)
            instance_count: Number of instances (default: 1)
            model_path: Optional path to model directory
        """
        self.env_folder = Path(env_folder).resolve()
        self.scoring_folder = Path(scoring_folder).resolve()
        self.model_path = Path(model_path) if model_path else None
        
        # Configuration
        self.endpoint_name = endpoint_name or self.DEFAULT_ENDPOINT_NAME
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model_version = model_version or self.DEFAULT_MODEL_VERSION
        self.env_name = env_name or self.DEFAULT_ENV_NAME
        self.env_version = env_version or self.DEFAULT_ENV_VERSION
        self.deployment_name = deployment_name or self.DEFAULT_DEPLOYMENT_NAME
        self.instance_type = instance_type or self.DEFAULT_INSTANCE_TYPE
        self.instance_count = instance_count or self.DEFAULT_INSTANCE_COUNT
        
        self.ml_client = None
        self._validate_paths()
        self._authenticate()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.env_folder.exists():
            raise FileNotFoundError(f"Environment folder not found: {self.env_folder}")
        
        conda_file = self.env_folder / "conda.yaml"
        if not conda_file.exists():
            raise FileNotFoundError(f"conda.yaml not found in: {self.env_folder}")
        
        if not self.scoring_folder.exists():
            raise FileNotFoundError(f"Scoring folder not found: {self.scoring_folder}")
        
        score_file = self.scoring_folder / "score.py"
        if not score_file.exists():
            raise FileNotFoundError(f"score.py not found in: {self.scoring_folder}")
        
        if self.model_path and not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        logger.info("All required paths validated successfully")
    
    def _authenticate(self) -> None:
        """Authenticate with Azure ML workspace."""
        load_dotenv()
        
        # Get credentials
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if not all([tenant_id, client_id, client_secret]):
            raise EnvironmentError(
                "Missing required env vars for authentication: "
                "AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
            )
        
        # Get workspace info
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")
        
        missing = [name for name, val in [
            ("AZURE_SUBSCRIPTION_ID", subscription_id),
            ("AZURE_RESOURCE_GROUP", resource_group),
            ("AZUREML_WORKSPACE_NAME", workspace_name),
        ] if not val]
        
        if missing:
            raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
        
        # Create ML client
        credential = ClientSecretCredential(
            tenant_id=tenant_id,  # type: ignore
            client_id=client_id, # type: ignore
            client_secret=client_secret # type: ignore
        )
        
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        
        logger.info(f"Authenticated to workspace: {workspace_name}")
    
    def _download_insightface_model(self) -> Path:
        """
        Download InsightFace buffalo_l model if it doesn't exist locally.
        Downloads the model zip from GitHub releases and extracts it.
        
        Returns:
            Path to the extracted model directory (.insightface)
        """
        # Create cache directory
        cache_dir = Path.cwd() / "temporary" / "cache" / ".insightface" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_dir = cache_dir / "buffalo_l"
        
        # Check if model already exists
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Model already exists: {model_dir}")
            return Path.cwd() / "temporary" / "cache" / ".insightface"
        
        logger.info(f"Using InsightFace cache directory: {cache_dir}")
        
        # Model URL from GitHub
        url_to_download = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = model_dir / "buffalo_l.zip"
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the zip file
        logger.info(f"Downloading buffalo_l model from {url_to_download}")
        try:
            urllib.request.urlretrieve(url_to_download, str(zip_path))
            logger.info(f"Successfully downloaded to: {zip_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise
        
        # Extract the zip file
        logger.info(f"Extracting model to: {model_dir}")
        try:
            with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
                zip_ref.extractall(str(model_dir))
            logger.info(f"Successfully extracted model")
        except Exception as e:
            logger.error(f"Failed to extract model: {str(e)}")
            raise
        
        # Remove the zip file after extraction
        try:
            zip_path.unlink()
            logger.info(f"Cleaned up zip file: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to remove zip file: {str(e)}")
        
        return Path.cwd() / "temporary" / "cache" / ".insightface"
    
    def _try_get_model(self) -> Optional[Model]:
        """Try to get existing model from registry."""
        try:
            return self.ml_client.models.get( # type: ignore
                name=self.model_name,
                version=self.model_version
            )
        except ResourceNotFoundError:
            return None
    
    def _try_get_environment(self) -> Optional[Environment]:
        """Try to get existing environment."""
        try:
            return self.ml_client.environments.get( # type: ignore
                name=self.env_name,
                version=self.env_version
            ) 
        except ResourceNotFoundError:
            return None
    
    def _try_get_endpoint(self) -> Optional[ManagedOnlineEndpoint]:
        """Try to get existing endpoint."""
        try:
            return self.ml_client.online_endpoints.get(name=self.endpoint_name) # type: ignore
        except ResourceNotFoundError:
            return None
    
    def _try_get_deployment(self) -> Optional[ManagedOnlineDeployment]:
        """Try to get existing deployment."""
        try:
            return self.ml_client.online_deployments.get(
                name=self.deployment_name,
                endpoint_name=self.endpoint_name
            )
        except ResourceNotFoundError:
            return None
    
    def ensure_model(self, skip_if_exists: bool = True) -> Model:
        """
        Register or get the InsightFace model in Azure ML.
        
        Args:
            skip_if_exists: If True, skip creation if model already exists
            
        Returns:
            Model asset object
        """
        if skip_if_exists:
            existing = self._try_get_model()
            if existing is not None:
                logger.info(f"Model exists: {self.model_name}:{self.model_version}")
                return existing
        
        logger.info(f"Creating model: {self.model_name}:{self.model_version}")
        
        # Determine model path
        if self.model_path:
            model_path = str(self.model_path)
            logger.info(f"Using provided model path: {model_path}")
        else:
            # Download InsightFace model if not provided
            logger.info("No model path provided. Downloading InsightFace buffalo_l model...")
            model_path = self._download_insightface_model()
        
        model_asset = Model(
            name=self.model_name,
            version=self.model_version,
            path=model_path,
            description="InsightFace model for face recognition and analysis (buffalo_l)",
        )
        
        created = self.ml_client.models.create_or_update(model_asset)
        logger.info(f"Created model: {created.name}:{created.version}")
        return created
    
    def ensure_environment(self, skip_if_exists: bool = True) -> Environment:
        """
        Create or get the environment for InsightFace.
        
        Args:
            skip_if_exists: If True, skip creation if environment already exists
            
        Returns:
            Environment object
        """
        if skip_if_exists:
            existing = self._try_get_environment()
            if existing is not None:
                logger.info(f"Environment exists: {self.env_name}:{self.env_version}")
                return existing
        
        logger.info(f"Creating environment: {self.env_name}:{self.env_version}")
        
        conda_file = self.env_folder / "conda.yaml"
        
        env = Environment(
            name=self.env_name,
            version=self.env_version,
            description="Environment for InsightFace online endpoint",
            conda_file=str(conda_file),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
        
        created = self.ml_client.environments.create_or_update(env)
        logger.info(f"Created environment: {created.name}:{created.version}")
        return created
    
    def ensure_endpoint(self, skip_if_exists: bool = True) -> ManagedOnlineEndpoint:
        """
        Create or get the online endpoint.
        
        Args:
            skip_if_exists: If True, skip creation if endpoint already exists
            
        Returns:
            ManagedOnlineEndpoint object
        """
        if skip_if_exists:
            existing = self._try_get_endpoint()
            if existing is not None:
                logger.info(f"Endpoint exists: {self.endpoint_name}")
                return existing
        
        logger.info(f"Creating endpoint: {self.endpoint_name}")
        
        endpoint = ManagedOnlineEndpoint(
            name=self.endpoint_name,
            description="InsightFace scoring endpoint",
            auth_mode="key",
        )
        
        poller = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
        
        if isinstance(poller, LROPoller):
            result = poller.result()
            logger.info(f"Created endpoint: {result.name}")
            return result
        
        return poller
    
    def ensure_deployment(
        self,
        model: Model,
        environment: Environment,
        skip_if_exists: bool = True
    ) -> ManagedOnlineDeployment:
        """
        Create or get the deployment.
        
        Args:
            model: Model asset to deploy
            environment: Environment to use
            skip_if_exists: If True, skip creation if deployment already exists
            
        Returns:
            ManagedOnlineDeployment object
        """
        if skip_if_exists:
            existing = self._try_get_deployment()
            if existing is not None:
                logger.info(
                    f"Deployment exists: {self.deployment_name} "
                    f"(endpoint={self.endpoint_name})"
                )
                return existing
        
        logger.info(
            f"Creating deployment: {self.deployment_name} "
            f"(endpoint={self.endpoint_name})"
        )
        
        deployment = ManagedOnlineDeployment(
            name=self.deployment_name,
            endpoint_name=self.endpoint_name,
            model=model,
            environment=environment,
            code_configuration=CodeConfiguration(
                code=str(self.scoring_folder),
                scoring_script="score.py",
            ),
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            request_settings=OnlineRequestSettings(
                max_concurrent_requests_per_instance=1,
                request_timeout_ms=60000,
            ),
            app_insights_enabled=True,
        )
        
        poller = self.ml_client.online_deployments.begin_create_or_update(deployment)
        
        if isinstance(poller, LROPoller):
            logger.info(f"Waiting for deployment '{self.deployment_name}' to complete...")
            result = poller.result()
            logger.info(f"Created deployment: {result.name}")
            return result
        
        return poller
    
    def set_traffic(self, deployment_name: Optional[str] = None) -> None:
        """
        Route all traffic to the deployment.
        
        Args:
            deployment_name: Deployment to route traffic to (default: blue)
        """
        deployment_name = deployment_name or self.deployment_name
        
        endpoint = self.ml_client.online_endpoints.get(name=self.endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Traffic for {self.endpoint_name} set to: {endpoint.traffic}")
    
    def get_endpoint_credentials(self) -> tuple:
        """
        Get the scoring URI and API key for the endpoint.
        
        Returns:
            Tuple of (scoring_uri, api_key)
        """
        endpoint = self.ml_client.online_endpoints.get(name=self.endpoint_name)
        keys = self.ml_client.online_endpoints.get_keys(name=self.endpoint_name)
        
        return endpoint.scoring_uri, keys.primary_key
    
    def save_credentials_to_env(self, env_file: Optional[str] = None) -> None:
        """
        Save endpoint credentials to .env file.
        
        Args:
            env_file: Path to .env file (default: find in current directory)
        """
        scoring_uri, api_key = self.get_endpoint_credentials()
        
        if env_file is None:
            env_file = find_dotenv()
        
        if not env_file:
            env_file = ".env"
        
        set_key(env_file, "INSIGHTFACE_ENDPOINT_URI", scoring_uri)
        set_key(env_file, "INSIGHTFACE_ENDPOINT_KEY", api_key)
        
        logger.info(f"Saved credentials to {env_file}")
        logger.info(f"Endpoint URI: {scoring_uri}")
    
    def deploy(self, save_credentials: bool = True) -> None:
        """
        Execute the full deployment pipeline.
        
        Args:
            save_credentials: If True, save endpoint credentials to .env file
        """
        logger.info("=" * 60)
        logger.info("Starting InsightFace Model Deployment")
        logger.info("=" * 60)
        
        try:
            # Step 1: Ensure model
            logger.info("\n[Step 1/5] Ensuring model asset...")
            model = self.ensure_model()
            
            # Step 2: Ensure environment
            logger.info("\n[Step 2/5] Ensuring environment...")
            environment = self.ensure_environment()
            
            # Step 3: Ensure endpoint
            logger.info("\n[Step 3/5] Ensuring endpoint...")
            endpoint = self.ensure_endpoint()
            
            # Step 4: Ensure deployment
            logger.info("\n[Step 4/5] Ensuring deployment...")
            deployment = self.ensure_deployment(model, environment)
            
            # Step 5: Configure traffic
            logger.info("\n[Step 5/5] Configuring traffic...")
            self.set_traffic()
            
            # Save credentials
            if save_credentials:
                logger.info("\nSaving credentials to .env...")
                self.save_credentials_to_env()
            
            logger.info("\n" + "=" * 60)
            logger.info("Deployment completed successfully!")
            logger.info("=" * 60)
            
            # Print endpoint info
            scoring_uri, api_key = self.get_endpoint_credentials()
            logger.info(f"\nEndpoint: {self.endpoint_name}")
            logger.info(f"Scoring URI: {scoring_uri}")
            logger.info(f"API Key: {api_key[:10]}... (truncated)")
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy InsightFace model to Azure Machine Learning Online Endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deployment
  python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript
  
  # With custom endpoint name
  python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript \\
                   --endpoint-name my-insightface-endpoint
  
  # With custom names
  python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript \\
                   --endpoint-name my-endpoint --model-name my-insightface
  
  # Custom instance type
  python deploy.py --env-folder ./Environment --scoring-folder ./ScoringScript \\
                   --instance-type Standard_D4s_v3 --instance-count 2
        """
    )
    
    parser.add_argument(
        "--env-folder",
        required=True,
        help="Path to environment folder containing conda.yaml"
    )
    parser.add_argument(
        "--scoring-folder",
        required=True,
        help="Path to scoring script folder containing score.py"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to model directory"
    )
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help=f"Azure ML endpoint name (default: {AzureMLInsightFaceDeployment.DEFAULT_ENDPOINT_NAME})"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"Model registry name (default: {AzureMLInsightFaceDeployment.DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=f"Model version (default: {AzureMLInsightFaceDeployment.DEFAULT_MODEL_VERSION})"
    )
    parser.add_argument(
        "--env-name",
        default=None,
        help=f"Environment name (default: {AzureMLInsightFaceDeployment.DEFAULT_ENV_NAME})"
    )
    parser.add_argument(
        "--env-version",
        default=None,
        help=f"Environment version (default: {AzureMLInsightFaceDeployment.DEFAULT_ENV_VERSION})"
    )
    parser.add_argument(
        "--deployment-name",
        default=None,
        help=f"Deployment name (default: {AzureMLInsightFaceDeployment.DEFAULT_DEPLOYMENT_NAME})"
    )
    parser.add_argument(
        "--instance-type",
        default=None,
        help=f"Compute instance type (default: {AzureMLInsightFaceDeployment.DEFAULT_INSTANCE_TYPE})"
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=None,
        help=f"Number of instances (default: {AzureMLInsightFaceDeployment.DEFAULT_INSTANCE_COUNT})"
    )
    parser.add_argument(
        "--no-save-credentials",
        action="store_true",
        help="Don't save endpoint credentials to .env file"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = AzureMLInsightFaceDeployment(
            env_folder=args.env_folder,
            scoring_folder=args.scoring_folder,
            endpoint_name=args.endpoint_name,
            model_name=args.model_name,
            model_version=args.model_version,
            env_name=args.env_name,
            env_version=args.env_version,
            deployment_name=args.deployment_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            model_path=args.model_path,
        )
        
        deployer.deploy(save_credentials=not args.no_save_credentials)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()