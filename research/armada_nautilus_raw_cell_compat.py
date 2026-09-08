"""Nautilus catalog compatibility entrypoint for Armada clean-room BT."""
from nautilus_catalog_compat import install_query_quote_ticks_alias

install_query_quote_ticks_alias()

from armada_nautilus_raw_cell import main

if __name__ == "__main__":
    main()
