import argparse
import re
from persona.evaluators.base import BaseEvaluator

class MMLUEvaluator(BaseEvaluator):
    
    def _extract_answer(self, prediction: str):
        if not prediction:
            return None

        # Strip <think> tags if present
        text = prediction.replace("<think>", "").replace("</think>", "").strip()

        # Normalize whitespace
        text = " ".join(text.split())

        # Combined regex for all common answer formats
        match = re.search(
            r'"answer"\s*:\s*"([A-Da-d])"'      # JSON format
            r'|answer\s*(?:is)?\s*:?\s*\(?([A-Da-d])\)?[.,]?',  # answer: C / answer is C
            text,
            re.IGNORECASE
        )
        if match:
            return (match.group(1) or match.group(2)).upper()

        # Fallback: Last standalone A-D in text
        matches = re.findall(r'\b([A-Da-d])\b', text)
        if matches:
            return matches[-1].upper()

        return None

    def _normalize(self, text):
        if not text:
            return ""
        return text.replace("(", "").replace(")", "").upper().strip()

    def _check_equal(self, instance) -> bool:
        '''Compare prediction against the reference'''
        gt = self._normalize(instance.get('answer', ''))
        pred = self._normalize(instance.get('predicted_answer', ''))

        return gt == pred


parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', help='Path to the file with the predictions', type=str)
parser.add_argument('--output_path', help='Path for the output file', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    evaluator = MMLUEvaluator()
    evaluator.evaluate(preds_path=args.eval_path, out_path=args.output_path)