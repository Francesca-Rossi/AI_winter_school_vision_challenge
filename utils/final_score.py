import argparse
import json

from utils.docvqa_eval.docvqa_test import compute_score_docvqa
from utils.amber_disc_eval.amber_disc_test import compute_amber_disc_score
from utils.evqa_eval.evqa_compute_metrics import compute_score_evqa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_evqa", type=str, default=None)
    parser.add_argument("--input_path_docvqa", type=str, default=None)
    parser.add_argument("--input_path_amber_disc", type=str, default=None)

    args = parser.parse_args()

    scores = {
        'evqa': compute_score_evqa(args.input_path_evqa) if args.input_path_evqa else 0.0,
        'docvqa': compute_score_docvqa(args.input_path_docvqa) if args.input_path_docvqa else 0.0,
        'amber_disc': compute_amber_disc_score(args.input_path_amber_disc) if args.input_path_amber_disc else 0.0
    }

    overall_score = (scores['evqa'] + scores['docvqa'] + scores['amber_disc']) / 3
    scores['overall'] = overall_score

    print(json.dumps(scores, indent=2))

# Esempio di come lo userai per valutare il loro JSON:
if __name__ == "__main__":
    main()
