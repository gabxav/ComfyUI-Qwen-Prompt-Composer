from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = './web'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Make every node in this package discoverable through the shared brand keyword.
for _node_class in set(NODE_CLASS_MAPPINGS.values()):
    _search_aliases = list(getattr(_node_class, "SEARCH_ALIASES", []))
    if "xavierlab" not in [alias.casefold() for alias in _search_aliases]:
        _node_class.SEARCH_ALIASES = [*_search_aliases, "xavierlab"]
del _node_class, _search_aliases
