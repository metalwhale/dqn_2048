FROM nvcr.io/nvidia/tensorflow:20.06-tf2-py3

RUN apt-get update -y

# Configure timezone. See: https://askubuntu.com/a/1013396
ENV TZ=Asia/Tokyo DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Arguments
ARG uname
ARG gname=dqn
ARG uid
ARG gid

# Add current user from host
RUN groupadd -g $gid $gname \
    && useradd -u $uid -g $gname -r $uname \
    && mkdir /home/$uname \
    && chown $uid:$gid -R /home/$uname

# Setup environment
USER $uname
WORKDIR /home/$uname/dqn_2048
COPY requirements.txt ./
RUN pip install -U -r requirements.txt
