from pathlib import Path
import yaml
import litellm
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

# Enable dropping unsupported parameters (e.g., temperature/top_p for reasoning models)
litellm.drop_params = True

_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

# Shared across all callers — models are stateless so this is safe.
_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        config = yaml.safe_load(_CONFIG_PATH.read_text())
        router_settings = config.get("router_settings", {})
        _router = Router(
            model_list=config["model_list"],
            fallbacks=router_settings.get("fallbacks", []),
        )
    return _router


def get_llm(role: str = "tool") -> ChatLiteLLMRouter:
    """Return a ChatLiteLLMRouter for the given role."""
    if role in ["retrieval-augment", "aggregator"]:
        # Enable thinking for some roles to enhance UX
        return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=True)
    return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=False)
