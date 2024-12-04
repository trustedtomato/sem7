import torch
from ail_parser import parse_intermixed_args_local


def modify_parser(parser):
    pass


def main(args):
    print(torch.__version__)
    print(torch.cuda.device_count())
    # do something on cuda 0
    x = torch.tensor([1, 2, 3]).cuda()
    x = x + 1
    print(x)


if __name__ == "__main__":
    main(parse_intermixed_args_local(modify_parser))

# This is a test script to check if torch is installed correctly
