from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from pointnav_agent import PointNavAgent
import numpy as np
from PIL import Image, ImageOps
import argparse
import torchvision.transforms as transforms
import time

app = Flask(__name__)
run_with_ngrok(app) 
possible_actions = ["Stop", "MoveForward", "TurnLeft", "TurnRight"]

mp3d_gibson_agent = PointNavAgent("models/trained/mp3d+gibson.pth", "data/configs/ppo_pointnav_gibson.yaml")
hm3d_agent = PointNavAgent("models/trained/hm3d.pth", "data/configs/ppo_pointnav_gibson.yaml")

@app.route("/")
def index():
    return "PointNav Server Online"

@app.route("/alive")
def alive():
    return "OK"

@app.route("/ready")
def ready():
    return jsonify(backend="ready")

@app.route("/mp3d+gibson", methods=["POST"])
def mp3d_gibson():
    rgb = get_rgb(request)
    depth = get_depth(request)
    pointgoal = get_pointgoal(request)
    observation = {"rgb": rgb, "depth": depth, "pointgoal_with_gps_compass": pointgoal}
    action = mp3d_gibson_agent.act([observation])
    return possible_actions[action]

@app.route("/hm3d", methods=["POST"])
def hm3d():
    rgb = get_rgb(request)
    depth = get_depth(request)
    pointgoal = get_pointgoal(request)
    observation = {"rgb": rgb, "depth": depth, "pointgoal_with_gps_compass": pointgoal}
    action = hm3d_agent.act([observation])
    return possible_actions[action]

@app.route("/reset")
def reset():
    mp3d_gibson_agent.reset()
    hm3d_agent.reset()
    return "Agents Reset" 

def get_pointgoal(request):
    distance = request.form["distance"]
    angle = request.form["angle"]
    pointgoal = np.array((distance, angle), dtype=np.float32)
    return pointgoal

def get_rgb(request):
    rgb = request.files["rgb"]
    rgb = Image.open(rgb)
    rgb = np.array(rgb)
    return rgb

def get_depth(request):
    depth = request.files["depth"]
    depth = Image.open(depth).convert("L")
    depth = np.array(depth, dtype=np.float32)
    depth = depth/255
    depth = np.expand_dims(depth, axis=2)
    return depth

app.run()