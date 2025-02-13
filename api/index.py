from app import app

# Vercel requires this handler
def handler(request, context):
    return app(request, context)
