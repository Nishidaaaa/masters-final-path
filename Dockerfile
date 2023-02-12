FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update
RUN apt-get install -y vim git

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user


ADD .ssh /home/user/.ssh
RUN chown -R user:user /home/user/.ssh
RUN chmod 600 /home/user/.ssh/id_rsa

ENV PATH $PATH:/home/user/.local/bin
ENV PYTHONPATH $PYTHONPATH:/home/user/.local/bin

USER user
COPY requirements.txt /tmp
RUN pip install --user -r /tmp/requirements.txt



