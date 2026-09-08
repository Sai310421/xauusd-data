"""Nautilus catalog compatibility entrypoint for G75 TSUGI BT."""
from nautilus_catalog_compat import install_query_quote_ticks_alias

install_query_quote_ticks_alias()

from g75_tsugi_nautilus_raw_bt import main

if __name__ == "__main__":
    main()
