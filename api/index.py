from middleware.app import app

# Vercel looks for `app` (ASGI) in api/index.py
__all__ = ["app"]
