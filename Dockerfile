FROM python:3.10-slim
COPY ml /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install libgomp1 -y
CMD ["python", "app.py"]