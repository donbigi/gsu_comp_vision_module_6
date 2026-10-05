# Optical flow + structure from motion demo — serves precomputed results.
# No OpenCV/numpy at runtime: the pipeline (synthetic_data.py -> optical_flow.py
# -> structure_from_motion.py) is run offline by generate_results.py, which bakes
# the figures, flow animations and results.json into static/. This image only
# serves those files, so it stays small and never runs heavy compute in the pod.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Flask is the only runtime dependency.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (templates/, static/, app.py) plus the precomputed
# results already inside static/.
COPY . .

# Run as an unprivileged user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "app.py"]
