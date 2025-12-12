# Use Triton Python-enabled image. If nvcr.io pull needs NGC login you can use a DockerHub mirror tag.
FROM hubimage/nvcr-io-nvidia-tritonserver:25.05-py3

# (Optional) install OS packages
RUN apt-get update && apt-get install -y git build-essential

# Install Python packages needed by your model and the Triton Python client
# Note: install torch + sentence-transformers (CPU) and triton client libraries
RUN python3 -m pip install \
    transformers \
    accelerate \
    numpy \
    tritonclient[http]

# Expose Triton default ports (optional)
EXPOSE 8000 8001 8002

# Default command will be overridden when running; keep an entrypoint for convenience
ENTRYPOINT ["tritonserver"]
