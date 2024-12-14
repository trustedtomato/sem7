import os

import numpy as np
import pandas as pd
import sacrebleu
import scipy.stats as st
from aac_metrics import Evaluate
from rouge_score import rouge_scorer
from tqdm import tqdm


class TSCapMetrics:
    def __init__(self, java_path: str = "~/jdk-11.0.2/bin/java"):
        self.evaluator = Evaluate(
            metrics=[
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "meteor",
                "cider_d",
            ],
            java_path=java_path,
        )

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        self._metrics = {}

    def _add_score_to_metrics(self, metric_name, score):
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(score)

    # Function to calculate sacreBLEU score
    def _calculate_bleu_scores(self, ref_sentences, hyp_sentences):
        sacre_bleu_score = sacrebleu.corpus_bleu(hyp_sentences, [ref_sentences])
        self._add_score_to_metrics("sacreBLEU", sacre_bleu_score.score / 100)

    # Function to calculate ROUGE score
    def _calculate_rouge_scores(self, ref_sentences, hyp_sentences):
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        for ref, hyp in zip(ref_sentences, hyp_sentences):
            scores = self.rouge_scorer.score(ref, hyp)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)
        for key, values in zip(
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            [rouge1_scores, rouge2_scores, rougeL_scores],
        ):
            self._add_score_to_metrics(key, np.mean(values))

    # Function to calculate CIDEr score
    def _calculate_aac_scores(self, ref_sentences, hyp_sentences):
        ref_sentences_as_singletons = [[sentence] for sentence in ref_sentences]
        metrics, _ = self.evaluator(hyp_sentences, ref_sentences_as_singletons)
        for key, value in metrics.items():
            self._add_score_to_metrics(key, value.item())

    def calculate_metrics(
        self,
        ref_sentences: list[str],
        hyp_sentences: list[list[str]],
        test_loss: float,
        out_dir: str | None = None,
        out_name: str | None = None,
    ):
        # hyp_sentences should be a list of list of strings where the dimensions are [n_samples, n_sentences]
        # So you would run predict n_samples amount of times and store the result of predict in a list of size n_sentences
        # ref_sentences should be a list of strings where the dimensions are [n_sentences] and contains the ground truth sentences
        # See the example in main
        metrics_with_conf_int = {}
        for i in tqdm(range(len(hyp_sentences))):
            self._calculate_bleu_scores(ref_sentences, hyp_sentences[i])
            self._calculate_rouge_scores(ref_sentences, hyp_sentences[i])
            self._calculate_aac_scores(ref_sentences, hyp_sentences[i])

        for key, value in self._metrics.items():
            conf_int = st.t.interval(
                0.95, len(value) - 1, loc=np.mean(value), scale=st.sem(value)
            )
            metrics_with_conf_int[key] = {
                "mean": float(round(np.mean(value), 3)),
                "confidence_interval": (
                    float(round(conf_int[0], 3)),
                    float(round(conf_int[1], 3)),
                ),
            }
        if out_dir and out_name:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            metrics_df = pd.DataFrame(metrics_with_conf_int)
            metrics_df.insert(0, "test_loss", round(test_loss, 3))
            metrics_df.to_csv(os.path.join(out_dir, out_name), index=False)

        return metrics_with_conf_int


if __name__ == "__main__":
    # Example predictions and references
    predictions = [["hello general kenobi, how are you", "foo bar bar bar foobar"]] * 10
    references = [
        "hello there general kenobi, how are you",
        "foo bar bar bar bar foobar",
    ]
    # predictions = [["Hello there my friend", "Apples are red"]]
    # references = ["Hello there my friend", "Apples are yellow"]
    # Initialize the TSCapMetrics class
    ts_cap_metrics = TSCapMetrics()

    print(
        ts_cap_metrics.calculate_metrics(
            references,
            predictions,
            1,
            out_dir="./metrics",
            out_name="test.csv",
        )
    )
