import sys
import argparse
import pdb
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaOverlay, cudaDeviceSynchronize
import uvicorn


# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 
parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--depthnetwork", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

app = FastAPI()

origins = [
	"https://localhost",
	"http://localhost",
	"https://play.unity.com",
    "*"
	]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
	)

net = poseNet(args.network, sys.argv, args.threshold)


args.input = "/dev/video0"

input = videoSource(args.input, argv=sys.argv)


@app.get('/')
def index():
    time_start = time.time()
    eye_location = {}
    eye_location["1"] = {}
    eye_location["2"] = {}
    img = input.Capture()

    if img is None: 
        return "No Image"
    poses = net.Process(img, overlay=args.overlay)
    eye_location = get_eyes(eye_location, poses)

    time_end = time.time()
    print(time_end - time_start)
    return eye_location

def get_eyes(eye_location, poses):
    for pose in poses:
        for keypoint in pose.Keypoints:
             if keypoint.ID == 1 or keypoint.ID == 2:
                eye_location[str(keypoint.ID)]["x"] = str(keypoint.x)
                eye_location[str(keypoint.ID)]["y"] = str(keypoint.y)
    return eye_location


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
