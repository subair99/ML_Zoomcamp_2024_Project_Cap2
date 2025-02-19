#!/usr/bin/env python
# coding: utf-8

import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'brain_tumor_model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['inputs'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

def prepare_response(pb_response):
    preds = pb_response.outputs['output_0'].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)