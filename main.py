import argparse
from src.train import train_and_evaluate
from src.detect import predict_file
from src.visualize import create_graphs
from src.simulate import create_demo

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Cybersecurity Threat Detection System"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train model and evaluate on official test split")
    sub.add_parser("visualize", help="Create evaluation and prediction graphs")
    sim = sub.add_parser("simulate", help="Create an offline sample from the test dataset")
    sim.add_argument("--rows", type=int, default=100)
    detect = sub.add_parser("detect", help="Predict records from a compatible CSV")
    detect.add_argument("input_csv")
    detect.add_argument("--output")
    args = parser.parse_args()

    if args.command == "train":
        train_and_evaluate()
    elif args.command == "visualize":
        create_graphs()
    elif args.command == "simulate":
        create_demo(rows=args.rows)
    elif args.command == "detect":
        predict_file(args.input_csv, args.output)

if __name__ == "__main__":
    main()
