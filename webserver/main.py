from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from pointnav_agent import PointNavAgent
import numpy as np
from PIL import Image, ImageOps
import argparse
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser(description="Webserver")
parser.add_argument("--model", type=str, help="Model Path", dest="model", default="models/trained/hm3d.pth")
parser.add_argument("--config", type=str, help="Config Path", dest="config", default="data/configs/ppo_pointnav_hm3d.yaml")
parser.add_argument("--mode", type=str, help="Input Mode (rgb, depth, rgbd)", dest="mode", default="rgbd")
args = parser.parse_args()

app = Flask(__name__)
run_with_ngrok(app) 
possible_actions = ["Stop", "MoveForward", "TurnLeft", "TurnRight"]

#rgb_agent = PointNavAgent("models/trained/rgb-10mio.pth", "data/configs/ppo_pointnav_mp3d.yaml", "rgb")
#depth_agent = PointNavAgent("models/trained/depth-10mio.pth", "data/configs/ppo_pointnav_mp3d.yaml", "depth")
rgb_depth_agent = PointNavAgent(args.model, args.config, args.mode)

@app.route("/")
def index():
    return "PointNav Server Online"

@app.route("/alive")
def alive():
    return "OK"

@app.route("/ready")
def ready():
    return jsonify(backend="ready")

# @app.route("/rgb", methods=["POST"])
# def rgb():
#     rgb = get_rgb(request)
#     pointgoal = get_pointgoal(request)
#     action = rgb_agent.act({"rgb": rgb, "pointgoal_with_gps_compass": pointgoal})
#     return possible_actions[action]

# @app.route("/depth", methods=["POST"])
# def depth():
#     depth = get_depth(request)
#     pointgoal = get_pointgoal(request)
#     action = depth_agent.act({"depth": depth, "pointgoal_with_gps_compass": pointgoal})["action"]
#     return possible_actions[action]

@app.route("/rgb-depth", methods=["POST"])
def rgb_depth():
    rgb = get_rgb(request)
    depth = get_depth(request)
    pointgoal = get_pointgoal(request)
    observation = {"rgb": rgb, "depth": depth, "pointgoal_with_gps_compass": pointgoal}
    action = rgb_depth_agent.act([observation])
    return possible_actions[action]

@app.route("/reset")
def reset():
    # rgb_agent.reset()
    # depth_agent.reset()
    rgb_depth_agent.reset()
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