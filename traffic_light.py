# Import Important Libraries

from scipy import misc
import io
from urllib.request import urlopen
import urllib
from dash.exceptions import PreventUpdate
import base64
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import logging
import datetime
import dash
import six

import matplotlib.pyplot as plt
import functools
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
from PIL import Image
from os import path

import time
import cv2
import collections


from utils import *
from darknet import Darknet

logger = logging.getLogger(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Set the location and name of the cfg file
cfg_file = './cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/coco.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes


PLOTLY_LOGO = "https://previews.123rf.com/images/putracetol/putracetol1808/putracetol180800016/106096283-traffic-light-logo-icon-design.jpg"
search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),

    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, className='img_header')),
                    dbc.Col(dbc.NavbarBrand(
                        "traffic light detection", className="ml-4 header_title")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),

    ],
    color="primary",
    dark=True,
    className='header'
)

badges = html.Span(
    [
        dbc.Badge("or choose with", pill=True,
                  color="light", className="mr-1"), ], className='badges'
)


app.layout = html.Div(
    [
        navbar,


        html.Div([
            html.Div('object detection for traffic light with yolo 3',
                     className='desciption_title'),
            dbc.Row([



                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Upload Image ',

                    ]),
                    className='upload_img',

                    # Allow multiple files to be uploaded
                    multiple=True
                ),

                dbc.Button("Predict", color="primary",
                           className="mr-1 button", id='show-secret',
                           ),
            ], className='test'),

            dbc.Row([
                html.Div(id='output-container-button'),
                html.Div(id='output-image-upload'), ], style={"width": '80%'}),
            html.Div(id='body-div')


        ], className='container_layout')

    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output1(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output(component_id='body-div', component_property='children'),
    [Input(component_id='show-secret', component_property='n_clicks')]

)
def update_output3(n_clicks):
    if n_clicks:
        class_names = load_class_names(namesfile)

    # Set the default figure size
        plt.rcParams['figure.figsize'] = [24.0, 14.0]

    # Load the image
        img = cv2.imread('test11.png')

        # Convert the image to RGB
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # We resize the image to the input width and height of the first layer of the network.
        resized_image = cv2.resize(img, (m.width, m.height))

        iou_thresh = 0.4
        nms_thresh = 0.6

        boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
        # print_objects(boxes, class_names)
        boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
        # params = with_height(original_image, boxes, class_names, plot_labels=True)

        plot_boxes(original_image, boxes, class_names, plot_labels=True)
        # print(a)
        # image_nps = image_nps[0][0]


def parse_contents(contents, filename, date):
    value = contents.split(',')
    value = value[1]
    image = base64.b64decode("{}".format(value))
    img = Image.open(io.BytesIO(image))
    img.save("test11.png", 'png')
    return html.Div([
        dbc.Card([

            dbc.CardBody(
                dbc.Row(
                    [dbc.Col(html.Div(
                        html.Img(src=contents, className='img_upload_pred'))),

                     ],
                    align="center",


                )),

        ], className='Card')
    ], className='container_predict')


if __name__ == '__main__':

    app.run_server(debug=True)
