from data_manager import results
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

results()