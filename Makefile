all: venv install run

venv:
	#########################################
	# init virtual environment
	#########################################
	python -m venv venv && . ./venv/bin/activate

install:
	#########################################
	# install dependencies
	#########################################
	pip install -r requirements.txt

run:
	#########################################
	# start staging server
	#########################################
	# python manage.py migrate
	python ./manage.py runserver 0.0.0.0:3000
