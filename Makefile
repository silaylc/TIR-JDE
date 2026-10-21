#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tir-jde
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Convert BSB into every model's format (EXP=experiment name)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m tir_jde.pipeline preprocess $(or $(EXP),strong_baseline_bsb)

## Train, evaluate and visualize one experiment (EXP=..., ARGS="seed=1 ...")
.PHONY: run
run:
	$(PYTHON_INTERPRETER) -m tir_jde.pipeline run $(EXP) $(ARGS)

## Compare experiments over seeds (EXPS="a b", SEEDS="0 1 2")
.PHONY: compare
compare:
	$(PYTHON_INTERPRETER) -m tir_jde.pipeline compare $(EXPS) --seeds $(or $(SEEDS),0)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
