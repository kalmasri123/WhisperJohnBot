FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Install Python
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y build-essential && \

    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y 
#  bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"
ADD ./ ./
RUN pip install -r requirements.txt
CMD [ "python3" ,"main.py" ]