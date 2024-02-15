FROM python:3.9
LABEL Pierre Augier <pierre.augier@univ-grenoble-alpes.fr>

RUN apt-get update

# also include utilities for debugging
RUN apt-get install -y --no-install-recommends \
  clang \
  emacs vim \
  meld less \
  # python3-pip is for Mercurial (hg-git, hg-evolve)
  mercurial python3-pip git
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

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN $(hg debuginstall -T '{pythonexe}') -m pip install hg-evolve hg-git --no-cache-dir --user

COPY --chown=appuser:appuser docker/hgrc $HOME/.hgrc
COPY --chown=appuser:appuser docker/pythranrc $HOME/.pythranrc

RUN mkdir -p $HOME/.local/include
RUN mkdir -p $HOME/.local/lib

ENV PATH=$HOME/.local/bin:$PATH

RUN mkdir -p $HOME/.config/matplotlib
RUN echo 'backend      : agg' > $HOME/.config/matplotlib/matplotlibrc

RUN python -m pip install -U pip pdm nox --user
