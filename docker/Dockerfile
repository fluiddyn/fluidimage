FROM python:3.11
LABEL Pierre Augier <pierre.augier@univ-grenoble-alpes.fr>
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update

# also include utilities for debugging
RUN apt-get install -y --no-install-recommends \
  clang \
  emacs vim \
  meld less git
RUN apt-get install -y --no-install-recommends \
  rsync \
  libgl1 xvfb xauth libgl1-mesa-dev \
  libxkbcommon-x11-0 \
  sudo graphviz

RUN rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && useradd -m -r -u 1000 -g appuser -s /bin/bash appuser -s /bin/bash && usermod -a -G sudo appuser
RUN echo appuser:appuser | chpasswd
USER appuser
ARG HOME=/home/appuser
RUN mkdir -p $HOME/opt
WORKDIR $HOME/opt
RUN echo $USER $HOME $PWD && whoami

COPY --chown=appuser:appuser docker/hgrc $HOME/.hgrc
COPY --chown=appuser:appuser docker/pythranrc $HOME/.pythranrc

RUN mkdir -p $HOME/.local/include
RUN mkdir -p $HOME/.local/lib

ENV PATH=$HOME/.local/bin:$PATH

RUN mkdir -p $HOME/.config/matplotlib
RUN echo 'backend      : agg' > $HOME/.config/matplotlib/matplotlibrc

RUN uv tool install mercurial --with hg-evolve,hg-git
RUN uv tool install pdm
RUN uv tool install nox
