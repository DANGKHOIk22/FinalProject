"""
Delete DeepFace Deployment from Azure Machine Learning

This script safely deletes the managed online endpoint and deployment to stop incurring costs.
It is idempotent and will gracefully handle resources that don't exist.

Safety Features:
    - Requires explicit confirmation via CONFIRM_DELETE environment variable or --force flag
    - Routes 0% traffic to deployment before deletion
    - Handles missing resources gracefully
    - Provides clear logging of all actions

Prerequisites:
    Same as deploy.py: Azure ML SDK credentials must be configured in .env

Usage:
    # Safe mode (will ask for confirmation)
    python delete_deployment.py --endpoint-name deep-face-endpoint-v2 \\
                                 --deployment-name blue
    
    # Force delete (requires --force flag)
    python delete_deployment.py --endpoint-name deep-face-endpoint-v2 \\
                                 --deployment-name blue --force
    
    # Using environment variable
    export CONFIRM_DELETE=true
    python delete_deployment.py --endpoint-name deep-face-endpoint-v2 \\
                                 --deployment-name blue
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Azure ML SDK imports
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from typing import Optional
from azure.core.exceptions import ResourceNotFoundError

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


class AzureMLDeepFaceDelete:
    """Handles deletion of DeepFace deployment from Azure ML."""
    
    # Default configuration
    DEFAULT_ENDPOINT_NAME = "deep-face-endpoint"
    DEFAULT_DEPLOYMENT_NAME = "blue"
    
    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ):
        """
        Initialize deletion handler.
        
        Args:
            endpoint_name: Azure ML endpoint name (default: deep-face-endpoint-v2)
            deployment_name: Deployment name to delete (default: blue)
        """
        self.endpoint_name = endpoint_name or self.DEFAULT_ENDPOINT_NAME
        self.deployment_name = deployment_name or self.DEFAULT_DEPLOYMENT_NAME
        
        self.ml_client = None
        self._authenticate()
    
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
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        
        logger.info(f"Authenticated to workspace: {workspace_name}")
    
    def _endpoint_exists(self) -> bool:
        """Check if endpoint exists."""
        try:
            self.ml_client.online_endpoints.get(name=self.endpoint_name)
            return True
        except ResourceNotFoundError:
            return False
    
    def _deployment_exists(self) -> bool:
        """Check if deployment exists."""
        try:
            self.ml_client.online_deployments.get(
                name=self.deployment_name,
                endpoint_name=self.endpoint_name
            )
            return True
        except ResourceNotFoundError:
            return False
    
    def _route_to_zero_traffic(self) -> None:
        """Route 0% traffic to deployment before deletion."""
        try:
            if not self._endpoint_exists():
                logger.warning(
                    f"Endpoint '{self.endpoint_name}' does not exist. Skipping traffic routing."
                )
                return
            
            logger.info(
                f"Routing 0% traffic to deployment '{self.deployment_name}' "
                f"(endpoint='{self.endpoint_name}')..."
            )
            
            endpoint = self.ml_client.online_endpoints.get(name=self.endpoint_name)
            
            # Set traffic to 0 for the deployment we're about to delete
            if self.deployment_name in endpoint.traffic:
                endpoint.traffic[self.deployment_name] = 0
            
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info("Traffic successfully routed to 0%")
            
        except Exception as e:
            logger.warning(
                f"Failed to route traffic to 0%: {str(e)}. "
                f"Proceeding with deletion anyway."
            )
    
    def delete_deployment(self) -> bool:
        """
        Delete the deployment.
        
        Returns:
            True if deletion succeeded, False if resource not found
        """
        try:
            if not self._deployment_exists():
                logger.info(
                    f"Deployment '{self.deployment_name}' "
                    f"(endpoint='{self.endpoint_name}') does not exist."
                )
                return False
            
            logger.info(
                f"Deleting deployment '{self.deployment_name}' "
                f"from endpoint '{self.endpoint_name}'..."
            )
            
            self.ml_client.online_deployments.begin_delete(
                name=self.deployment_name,
                endpoint_name=self.endpoint_name,
            ).result()
            
            logger.info(f"Successfully deleted deployment '{self.deployment_name}'")
            return True
            
        except ResourceNotFoundError:
            logger.info(f"Deployment '{self.deployment_name}' not found (already deleted)")
            return False
        except Exception as e:
            logger.error(f"Failed to delete deployment: {str(e)}", exc_info=True)
            raise
    
    def delete_endpoint(self) -> bool:
        """
        Delete the entire endpoint (after all deployments are removed).
        
        Returns:
            True if deletion succeeded, False if resource not found
        """
        try:
            if not self._endpoint_exists():
                logger.info(f"Endpoint '{self.endpoint_name}' does not exist.")
                return False
            
            logger.info(f"Deleting endpoint '{self.endpoint_name}'...")
            
            self.ml_client.online_endpoints.begin_delete(
                name=self.endpoint_name
            ).result()
            
            logger.info(f"Successfully deleted endpoint '{self.endpoint_name}'")
            return True
            
        except ResourceNotFoundError:
            logger.info(f"Endpoint '{self.endpoint_name}' not found (already deleted)")
            return False
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {str(e)}", exc_info=True)
            raise
    
    def cleanup(self, delete_endpoint: bool = True) -> None:
        """
        Execute the cleanup pipeline (deployment + optional endpoint deletion).
        
        Args:
            delete_endpoint: If True, also delete the entire endpoint
        """
        logger.info("=" * 60)
        logger.info("Starting DeepFace Deployment Cleanup")
        logger.info("=" * 60)
        
        try:
            # Step 1: Route traffic to 0%
            logger.info("\n[Step 1/2] Routing traffic to 0%...")
            self._route_to_zero_traffic()
            
            # Step 2: Delete deployment
            logger.info("\n[Step 2/2] Deleting deployment...")
            self.delete_deployment()
            
            # Optional Step 3: Delete endpoint
            if delete_endpoint:
                logger.info("\n[Step 3/3] Deleting endpoint...")
                self.delete_endpoint()
            
            logger.info("\n" + "=" * 60)
            logger.info("Cleanup completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
            raise


def ask_confirmation(force: bool = False) -> bool:
    """
    Ask user for confirmation to proceed with deletion.
    
    Args:
        force: If True, require explicit --force flag (skip interactive prompt)
        
    Returns:
        True if user confirms, False otherwise
    """
    # Check environment variable first
    if os.getenv("CONFIRM_DELETE", "").lower() == "true":
        logger.info("Deletion confirmed via CONFIRM_DELETE environment variable")
        return True
    
    # Check force flag
    if force:
        logger.info("Deletion confirmed via --force flag")
        return True
    
    # Ask interactively
    logger.warning("\n" + "!" * 60)
    logger.warning("WARNING: This will delete the deployment and endpoint!")
    logger.warning("This action cannot be undone.")
    logger.warning("!" * 60)
    
    response = input("\nType 'DELETE' to confirm deletion: ").strip().upper()
    
    if response == "DELETE":
        logger.info("User confirmed deletion")
        return True
    else:
        logger.info("User cancelled deletion")
        return False


def main():
    """Main entry point for the deletion script."""
    parser = argparse.ArgumentParser(
        description="Delete DeepFace deployment from Azure Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will ask for confirmation)
  python delete_deployment.py
  
  # With custom endpoint/deployment names
  python delete_deployment.py --endpoint-name my-endpoint --deployment-name blue
  
  # Force delete without confirmation
  python delete_deployment.py --force
  
  # Using CONFIRM_DELETE env variable
  export CONFIRM_DELETE=true
  python delete_deployment.py
  
  # Keep endpoint but delete deployment
  python delete_deployment.py --keep-endpoint
        """
    )
    
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help=f"Azure ML endpoint name (default: {AzureMLDeepFaceDelete.DEFAULT_ENDPOINT_NAME})"
    )
    parser.add_argument(
        "--deployment-name",
        default=None,
        help=f"Deployment name to delete (default: {AzureMLDeepFaceDelete.DEFAULT_DEPLOYMENT_NAME})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deletion without interactive confirmation"
    )
    parser.add_argument(
        "--keep-endpoint",
        action="store_true",
        help="Delete only the deployment, keep the endpoint"
    )
    
    args = parser.parse_args()
    
    try:
        # Ask for confirmation
        if not ask_confirmation(force=args.force):
            logger.info("Deletion cancelled by user")
            sys.exit(0)
        
        # Create deleter and execute cleanup
        deleter = AzureMLDeepFaceDelete(
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
        )
        
        deleter.cleanup(delete_endpoint=not args.keep_endpoint)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
