FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir "transformers>=4.40" fastapi "uvicorn[standard]" pydantic
WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8600
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8600","--log-level","warning"]
