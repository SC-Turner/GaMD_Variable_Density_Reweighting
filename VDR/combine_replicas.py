import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to combine gamd.log files with correct formatting for VDR")
    parser.add_argument("-g", "--gamd", help="gamd weights .log file locations as a list", required=False, nargs='+', default=[])
    parser.add_argument("-d", "--data", help="CV data text file locations as a list", required=False, nargs='+', default=[])
    args, leftovers = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    gamd_final = open('gamd_concat.log', 'a')
    data_final = open('data_concat.dat', 'a')
    print(args.gamd)
    print(args.data)
    for i in zip(args.gamd, args.data):
        gamd, data = i
        print(args.gamd[0])
        with open(gamd, 'r') as f:
            for line in f:
                if not line.startswith('  #'):
                    gamd_final.write(line)
        data_file = open(data, 'r')
        data_final.write(data_file.read())
        f.close()
        data_file.close()
    gamd_final.close()
    data_final.close()

if __name__=='__main__':
    main()
