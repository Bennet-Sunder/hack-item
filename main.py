import requests
import json
import yaml
from flask import Flask, redirect, url_for, request, render_template, Response
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa.nlu.model import Interpreter, Trainer
from rasa.model_training import train, train_nlu
import os
import tarfile
import rasa
import shutil


app = Flask(__name__)

dirname = os.path.dirname(__file__)
RASA_MODEL_PATH = os.path.join(dirname, "trained_models/nlu")
nlu_file =  os.path.join(dirname, 'data/nlu.yml')
config_file = os.path.join(dirname, 'config.yml')
out_dir = os.path.join(dirname, 'models')
trained_model_dir = os.path.join(dirname,'trained_models')

@app.route('/create_intent', methods = ['POST'])
def add_intent():
  input_intent = request.json
  yaml = YAML()
  yaml.preserve_quotes = True
  yaml.default_flow_style = False
  data = yaml.load(open(nlu_file, 'r'))
  print(input_intent)
  print('intent', input_intent['intent'])
  examples_text = '- ' + '\n- '.join(input_intent['examples']) + '\n'
  input_intent['examples'] = LiteralScalarString(examples_text)
  print(input_intent)
  data['nlu'].append(input_intent)
  yaml.dump(data, open(nlu_file, 'w'))

  file_path = train_nlu(config=config_file,
                        nlu_data=nlu_file, output=out_dir)
  print("Created MODEL ")
  print(file_path)
  for filename in os.listdir(trained_model_dir):
      rm_file_path = os.path.join(trained_model_dir, filename)
      try:
          if os.path.isfile(rm_file_path) or os.path.islink(rm_file_path):
              os.unlink(rm_file_path)
          elif os.path.isdir(rm_file_path):
              shutil.rmtree(rm_file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (rm_file_path, e))
  tar = tarfile.open(file_path, "r:gz")
  tar.extractall(path=trained_model_dir)
  tar.close()
  return Response({'success: true'}, status=200, mimetype='application/json')


@app.route('/predict', methods = ['POST'])
def predict():
  print("query", request.json['text'])
  intents = rasa.nlu.model.Interpreter.load(
            RASA_MODEL_PATH).parse(request.json['text'])
  print(intents)
  if intents['intent']['confidence'] > 0.90:
      data = { 'name': intents['intent']['name'], 'entities': intents['entities'] }
  print(data)
  return Response(json.dumps(data), status=200, mimetype='application/json')

if __name__ == '__main__':
  app.run(debug=True)
