FROM python:3.10-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5555
# ENV USER="Docker-User"
ENTRYPOINT [ "python" ]
CMD ["app.py"]
