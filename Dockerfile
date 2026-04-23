FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py .

ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache
ENV HUGGINGFACE_HUB_CACHE=/cache

CMD ["python", "handler.py"]