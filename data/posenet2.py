

import sys
import argparse
import pdb

import flask
app = flask.Flask(__name__)

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
#output = videoOutput(args.output, argv=sys.argv)

eye_location = {}
@app.route('/')
def index():
    img = input.Capture()

    if img is None: # timeout
        return "No Image"
    poses = net.Process(img, overlay=args.overlay)
    for pose in poses:
        for keypoint in pose.Keypoints:
             if keypoint.ID == 1 or keypoint.ID == 2:
                # pdb.set_trace()
                print(keypoint)
                id = str(keypoint.ID)
                eye_location[id] == {"x": keypoint.x, "y": keypoint.y}
    return eye_location

app.run(host="0.0.0.0", port="8050", debug=True)
