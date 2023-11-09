default: build

help:
	@echo 'Management commands for tricl:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the tricl project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t tricl 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=1"' --ipc=host --name tricl -v `pwd`:/workspace/tricl tricl:latest /bin/bash

up: build run

rm: 
	@docker rm tricl

stop:
	@docker stop tricl

reset: stop rm