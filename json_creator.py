import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xy',type=list, help='A list of [(xp1,yp1),(xp2,yp2)]')
parser.add_argument('--scatter', help='A list of [xp1,yp1,xp2,yp2]')
args = parser.parse_args()

if args.xy:
    print(args.xy[0][0])
    x1 = args.xy[0][0]
    y1 = args.xy[0][1]
    x2 = args.xy[1][0]
    y2 = args.xy[1][1]
data = {
    'ROI' :
        {
            'x1' : x1,
            'x2' : x2,
            'y1' : y1,
            'y2' : y2
        }
}


with open('json_data.json', 'w') as outfile:
    json.dump(data, outfile)