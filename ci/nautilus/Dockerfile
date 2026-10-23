FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
RUN python -m pip install --no-cache-dir 'nautilus_trader==1.230.0' pandas numpy
COPY . /workspace
CMD ["bash", "ci/nautilus/run_abc.sh"]
