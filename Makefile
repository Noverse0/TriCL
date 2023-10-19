default: build

help:
	@echo 'Management commands for sim_hgcl:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the sim_hgcl project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t sim_hgcl 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name sim_hgcl -v `pwd`:/workspace/sim_hgcl sim_hgcl:latest /bin/bash

up: build run

rm: 
	@docker rm sim_hgcl

stop:
	@docker stop sim_hgcl

reset: stop rm