import argparse
import re

from persona.evaluators.base import BaseEvaluator

class MMLUEvaluator(BaseEvaluator):
    
    def _extract_answer(self, prediction: str):
        if not prediction:
            return None

        # Normalize newlines for consistent regex matching
        text = prediction.strip()

        # Priority 1: Explicit JSON answer format {"answer": "C"}
        match = re.search(r'"answer"\s*:\s*"([A-Da-d])"', text)
        if match:
            return match.group(1).upper()

        # Priority 2: "answer is C" or "answer is: (C)"
        match = re.search(r'\banswer\s+is\s*:?\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Priority 3: "answer: C" or "answer: (C)"
        match = re.search(r'\banswer\s*:?\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Priority 4: A single capital letter on its own line
        match = re.search(r'^\s*([A-Da-d])\s*$', text, re.MULTILINE)
        if match:
            return match.group(1).upper()

        return None

    def _normalize(self, text):
        return text.replace("(", "").replace(")", "").upper().strip()

    def _check_equal(self, instance) -> bool:
        '''Compare prediction against the reference'''
        gt = self._normalize(instance['answer'])
        pred = self._normalize(instance['predicted_answer'])

        if gt == pred:
            return True
        else:
            return False


parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', help='Path to the file with the predictions', type=str)
parser.add_argument('--output_path', help='Path for the output file', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    evaluator = MMLUEvaluator()
    evaluator.evaluate(preds_path=args.eval_path, out_path=args.output_path)
