import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
# parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument("--one")
parser.add_argument("--two")
parser.add_argument("--three")

# args = parser.parse_args()
args = parser.parse_args()
print(args)