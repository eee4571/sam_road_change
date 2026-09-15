"""Experiment entry point; algorithms live in the production engine."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from engine.width.raw_width_reconstruction import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, ReconstructionConfig())
