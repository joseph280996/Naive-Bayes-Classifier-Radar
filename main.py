import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data", help="The path to boxes JSON file for testing")
parser.add_argument("-f", "--format", default=1, type=int, help="The format of the input file (1: JSON, 2: CSV)")

if __name__ == '__main__':
    args = parser.parse_args()
    file_data = getattr(args, 'data')
    file_format = getattr(args, 'format')
