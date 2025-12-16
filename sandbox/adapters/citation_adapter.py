import importlib
from typing import Callable, Optional, Iterable, Tuple, List, Dict, Any

from .path_setup import EXTERNAL_APP_ROOT  # noqa: F401 ensures side-effect sys.path append

# Lazily load external module to avoid import-time failures if path missing
_citation_utils = importlib.import_module("app.citation_utils")

parse_citations = _citation_utils.parse_citations
split_sentences = _citation_utils.split_sentences

# These helpers depend on a node_lookup abstraction; we keep the signature.
validate_citations = _citation_utils.validate_citations
CitationRef = _citation_utils.CitationRef
SentenceSpan = _citation_utils.SentenceSpan

# We also surface the inline regex pattern for ID harvesting.
CITATION_PATTERN = _citation_utils.CITATION_PATTERN


def render_links(text: str) -> str:
    """Convenience wrapper for render_citations_as_links."""
    return _citation_utils.render_citations_as_links(text)


def validate_summary_tool_output(tool_output: Dict[str, Any], node_lookup):
    return _citation_utils.validate_summary_tool_output(tool_output, node_lookup)
