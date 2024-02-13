import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to combine gamd.log files with correct formatting for VDR")
    parser.add_argument("-g", "--gamd", help="gamd weights .log file locations as a list", required=False, nargs='+', default=[])
    parser.add_argument("-d", "--data", help="CV data text file locations as a list", required=False, nargs='+', default=[])
    args, leftovers = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    if os.path.exists("GaMD_tempout.log"):
        os.remove("GaMD_tempout.log")
    if os.path.exists("gamd_concat.log"):
        os.remove("gamd_concat.log")
    if os.path.exists("data_concat.dat"):
        os.remove("data_concat.dat")
    for i in zip(args.gamd, args.data):
        gamd, data = i
        with open(gamd, 'r') as infile, open('GaMD_tempout.log', 'w') as outfile:
            for line in infile:
                if not line.startswith('  #'):
                    outfile.write(line)
        with open('GaMD_tempout.log', 'r') as infile:
            gamdlen = len(infile.readlines())
        with open(data, 'r') as infile:
            datalen = len(infile.readlines())
        min_entries = min(gamdlen, datalen)
        print(min_entries)
        print(gamd)
        with open('gamd_concat.log', 'a') as f, open('GaMD_tempout.log', 'r') as infile:
            d1 = infile.readlines()
            for i in range(min_entries):
                f.write(d1[i])
        with open('data_concat.dat', 'a') as f,  open(data, 'r') as infile:
            d2 = infile.readlines()
            for i in range(min_entries):
                f.write(d2[i])
    if os.path.exists("GaMD_tempout.log"):
        os.remove("GaMD_tempout.log")
