import logging
import os
import sys

import uvicorn

from src.config.settings import ConfigurationError, settings

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings.validate_llm()
    except ConfigurationError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        print("\nSet the required variables in .env and restart.", file=sys.stderr)
        sys.exit(1)

    try:
        uvicorn.run(
            "src.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            reload_dirs=["src"] if settings.debug else None,
            log_level=settings.log_level.lower(),
            # Cap the shutdown wait so Ctrl+C never hangs on a stuck thread.
            timeout_graceful_shutdown=1,
        )
    finally:
        os._exit(0)
