# Setup and Deployment Guide

## Docker Deployment

### Build the Docker image
```bash
docker build --platform linux/amd64 -t search-algorithm-api .
```

### Test locally using Lambda RIE (Runtime Interface Emulator)
```bash
docker run -p 9000:8080 \
  -e MISTRAL_API_KEY=your_api_key \
  -e USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
  search-algorithm-api
```

## Testing the API

```bash
curl -X POST \
  https://search-algo-iv563bk7p-apoorvs-projects-3369417a.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search: Hello hi"}'
```

## EC2 Deployment

### Connect to EC2
```bash
ssh -i "search-ec2.pem" ec2-user@ec2-43-204-115-205.ap-south-1.compute.amazonaws.com
```

### Clone Repository
```bash
git clone https://github.com/apoorvagni/search-algo-ai
cd search-algo-ai
```

### Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### Set Environment Variables
```bash
export MISTRAL_API_KEY=your_api_key_here
export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

### Server Management

#### Stop the server
```bash
kill -9 $(pgrep -f "gunicorn -w 4 -b 0.0.0.0:5000 app:app")
```

#### Start the server
```bash
nohup gunicorn -w 4 -b 0.0.0.0:5000 app:app &
```

This command starts the app with 4 worker processes, binding to all interfaces on port 5000. Adjust the number of workers and port as needed.

## TODO

### Production Improvements
- **Reverse Proxy (Optional)**: For production environments, consider using a reverse proxy like Nginx to handle incoming requests, manage SSL certificates, and forward requests to your Flask app. This setup can improve security and performance.
- **Rate Limiting**: Slow down the API calls to prevent rate limit issues
