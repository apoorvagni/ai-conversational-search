FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure environment variables are set
ENV PORT=3000
ENV MISTRAL_API_KEY=""
ENV USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

EXPOSE 3000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"] 