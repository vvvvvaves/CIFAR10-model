from torchviz import make_dot
import torch, os
from net import Net
from torchvision import transforms
from torchview import draw_graph

from data import get_data, get_norms


def add_to_classpath():
    path = "C:/Program Files/Graphviz/bin"
    os.environ["PATH"] += os.pathsep + path


def run():
    model_graph = draw_graph(Net(), input_size=(1, 3, 32, 32), expand_nested=True, save_graph=True, directory='./')
    model_graph.resize_graph(scale=5.0)  # scale as per the view
    model_graph.visual_graph.render(format='png')


add_to_classpath()
run()
