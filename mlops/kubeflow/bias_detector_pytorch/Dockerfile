FROM pytorch/pytorch:latest

RUN pip install Flask aif360 pandas flask-cors Minio Pillow torchsummary

ENV APP_HOME /app
COPY src $APP_HOME
WORKDIR $APP_HOME

ENTRYPOINT ["python"]
CMD ["app.py"]
