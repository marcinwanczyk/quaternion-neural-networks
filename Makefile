# AV/2023
SHELL := /usr/bin/bash
export LD_LIBRARY_PATH+=:/usr/lib/llvm-8/lib
# JETSON ORIN VARS
URL_START="https://developer.nvidia.com/embedded/learn"
URL_START+="/get-started-jetson-agx-orin-devkit"

# VENV PYTHON VARS
VENV_NAME=venv-$(shell hostname)-lpcvc
VENV_RUN=source $(VENV_NAME)/bin/activate
# lpcvai23/requirements.txt conflict in versions, relaxing
PY_REQUIREMENTS=requirements.txt
PY_RUN=file.py

# JETSON ORIN RULES
check-tegra:
	@echo Tegra, L4T version, flashed version of the BSP.
	@echo
# cat /etc/nv_tegra_release
#	type C:\path\to\nv_tegra_release
	@echo
	@echo "if REVISION < 1.0 update L4T repositories"
	@echo following $(URL_START)


update-tegra-repositories:
	sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/common r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'
	sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/t234 r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'

upgrade-distribution-system:
	sudo apt update
	sudo apt dist-upgrade
	@echo sudo reboot

install-jetpack-components: 
	sudo apt install nvidia-jetpack

install-venv-dependences:
	sudo apt install python3.8-venv

install-torch-dependences:
	sudo apt install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev

install-torchvision-dependences:
	sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev


TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl


# VENV PYTHON RULES

venv-create:
	python3 -m venv --clear $(VENV_NAME)

venv-install-packages:
	( \
	$(VENV_RUN) ; \
	python3 -m pip install  -r $(PY_REQUIREMENTS) ;\
	deactivate ; \
	)
TORCH_INSTALL=cache/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
venv-install-nvidia-torch:
	( \
	$(VENV_RUN)  ; \
	pip3 install --no-cache	$(TORCH_INSTALL); \
	deactivate ; \
	)

venv-remove:
	rm -rf $(VENV_NAME)
run:
	( \
	$(VENV_RUN) && ipython -i $(PY_RUN) ;\
	deactivate ; \
	)

run-ipython:
	( \
	$(VENV_RUN) && ipython  ;\
	deactivate ; \
	)
run-jupy:
	( \
	$(VENV_RUN) && jupyter lab  ;\
	deactivate ; \
	)

# LPCVAI
copy-sample:
	cp data/IMG/train/train_0001.png img_in/

create-solution:
	( \
	$(VENV_RUN) ; \
	python3 -m zipapp lpcvai23/solution -p='/usr/bin/env python' ;\
	deactivate ; \
	)

