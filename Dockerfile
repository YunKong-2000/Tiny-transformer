FROM nvcr.io/nvidia/pytorch:25.08-py3
WORKDIR /workspace/tiny-transformer
COPY . .
# Keep the image's CUDA-enabled PyTorch. Do not pip-install another torch wheel.
RUN python -m pip install --no-cache-dir -e '.[data]'
ENV PYTHONUNBUFFERED=1
CMD ["bash"]
