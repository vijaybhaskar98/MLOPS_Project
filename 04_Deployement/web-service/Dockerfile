FROM python:3.9-slim

RUN pip install -U pip #Update pip as certain packages (ex: XGBoost) need certain versions of pip
RUN pip install pipenv

WORKDIR /app #Creates and cd's into the /app directory

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy #--system installs the environment in the parent OS in the container
                                     #--deploy makes sure Pipfile.lock is up-to-date and will crash if it isn't

COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696 

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]