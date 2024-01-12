FROM tensorflow/tensorflow

WORKDIR /app

RUN apt update -y

run apt-get install git -y
COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt

COPY . .

EXPOSE 80

CMD [ "python", "-u", "main.py" ]
