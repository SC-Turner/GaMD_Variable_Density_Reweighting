import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Calculator for GaMD standard deviation limits")
    parser.add_argument("--frames", help="Total number of frames you plan to generate", required=True, type=int)
    parser.add_argument("--clusters", help="Number of local clusters you plan to generate, we recommemd at least 100", default=100, required=False, type=int)
    parser.add_argument("--stderr", help="Standard error of the boost potential standard deviation, we recommemd at a maximum of 0.02", default=0.02, required=False, type=float)
    parser.add_argument("--anharm", help="Anharmonicity tolerance of the boost potential standard deviation, we recommemd at a maximum of 0.01", default=0.01, required=False, type=float)
    args, leftovers = parser.parse_known_args()
    return args

def exponential(x, a, b, c):
    return a * np.exp(-b * x + c)

def return_x_exponential(y, a, b, c):
    return (np.log(y/a)-c)/-b

def calc_stddev(frames, clusters, stderr):
    per_cluster = frames/clusters
    stddev = stderr * (per_cluster ** 0.5)
    return stddev

def calc_req_frames_anharm(anharm_convergence):
    a = 1.6820718785282671
    b = 0.6865783403121103
    c = 1.1314585078623656
    yval = return_x_exponential(anharm_convergence, a, b, c)
    return a, b, c, np.exp(yval)

def calc_max_stddev(n, clusters, stderr, anharm):
    a, b, c, anharm_limit = calc_req_frames_anharm(anharm)
    stderr_limit = calc_stddev(n, clusters, stderr)
    min_frames = int(n/clusters)
    #print(min_frames)
    #print(anharm_limit)
    if min_frames < int(anharm_limit):
        print(f"Insufficient number of frames to converge the anharmonicity to below tolerance: {anharm}")
        print("")
        print(f"We recommend generating at least {int(anharm_limit)} frames per cluster, i.e. {int(anharm_limit)*100} total frames")
        print(f"Your current parameters will generate {int(min_frames)} frames per cluster")
        print(f"Please either generate more simulation frames, or utilise a higher anharmonicity tolerance with --anharm")
    else:
        print(f"Maximum standard deviation of the total boost potential: {stderr_limit}")

def main():
    args = parse_args()
    if args.anharm > 0.01:
        print("Warning: We cannot guarantee convergence of the reweighting error for anharmonicity tolerances above 0.01")
    if args.anharm < 0.01:
        print("Warning: Using an anharmonicity tolerance below 0.01 is unlikely to provide significant improvements to the reweighting accuracy, consider using a 0.01 anharmonicity tolerance")
    calc_max_stddev(args.frames, args.clusters, args.stderr, args.anharm)

if __name__ == '__main__':
    main()
