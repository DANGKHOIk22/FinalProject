from pathlib import Path
import yaml
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

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
    """Return a ChatLiteLLMRouter for the given role.

    Valid roles are defined in llm_config.yaml:
      planning   – Gemini 2.5 Flash, thinking_budget=3000
      execution  – Gemini 2.5 Flash Lite, thinking_budget=4000
      aggregator – Gemini 2.5 Flash, no thinking
      tool       – Gemini 2.5 Flash Lite, shared by tools / guardrails / query-understanding
    """
    if "role" in ["retrieval-augment", "aggregator"]:
        # Enable thinking for some roles to enhance UX
        return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=True)
    return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=False)
