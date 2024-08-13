import argparse
from model_utils import predict


def args_parser():
    parser = argparse.ArgumentParser(description="BCR-SORT")
    parser.add_argument("--input_file", "--i",default='/home/mist/BCR-SORT/data/cdr32.CSV',
                        help="Input file to predict cell subsets", type=str)
    parser.add_argument("--output_file", "--o", default='./cc07191.csv',
                        help="Output file containing prediction results of cell subsets", type=str)
    parser.add_argument("--model_path", "--p", default='/home/mist/BCR-SORT/result/best_wt.pt',
                        help="Path of model weights", type=str)
    parser.add_argument("--device", default=0,
                        help="GPU for prediction", type=int)

    return parser.parse_args()


def main():
    args = args_parser()
    predict(args)


if __name__ == "__main__":
    main()
