FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app
CMD ["python", "gradio_app.py"]