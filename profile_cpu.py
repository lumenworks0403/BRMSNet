"""Profile BRMSNet inference at the configured input resolution."""

import argparse
import logging

from models import MODEL_NAME, BRMSNet
from utils.training import profile_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    model = BRMSNet(pretrained=False).eval()
    summary = profile_model(model, args.image_size, logging.getLogger(__name__))
    print(f"Model: {MODEL_NAME} | Input: 1 x 3 x {args.image_size} x {args.image_size}")
    print(
        f"MACs: {summary['gmacs']:.4f} G | FLOPs: {summary['gflops']:.4f} G (2 per MAC)"
    )


if __name__ == "__main__":
    main()
