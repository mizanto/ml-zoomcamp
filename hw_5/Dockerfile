FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY ["src/bank.py", "./"]

COPY ["data/dv.bin", "data/model.bin", "./data/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "bank:app"]