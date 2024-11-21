import os

import nltk

nltk.download("wordnet")
import numpy as np
import pandas as pd
import ptbtok
import sacrebleu
import scipy.stats as st
import torch
from aac_metrics.functional import cider_d, spice
from bert_score.scorer import BERTScorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tqdm import tqdm


class TSCapMetrics:
    def __init__(self):
        # This tokenizer is default for sacrebleu, which is a common bleu scorer so we will use it for meteor too (mteval-v13a)
        self.tokenizer = Tokenizer13a()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading deberta for BERT score")
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli", device=device
        )
        print("Finished loading deberta")
        # We use the default tokenizer for ROUGE
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        self._metrics = {}

    # Function to calculate BLEU score
    def _calculate_bleu_scores(self, ref_sentences, hyp_sentences, max_n=4):
        bleu_sacre = sacrebleu.corpus_bleu(hyp_sentences, [ref_sentences])

        weights = [[1 / n for _ in range(n)] for n in range(1, max_n + 1)]
        ref_sentences_tokenized = [
            [self.tokenizer(ref_sentence).split(" ")] for ref_sentence in ref_sentences
        ]
        hyp_sentences_tokenized = [
            self.tokenizer(hyp_sentence).split(" ") for hyp_sentence in hyp_sentences
        ]
        # The BLEU-4 should be the same as sacrebleu assuming there's no need for smoothing I think
        bleu_nltk = corpus_bleu(
            ref_sentences_tokenized, hyp_sentences_tokenized, weights=weights
        )

        for i in range(1, max_n + 1):
            if f"BLEU-{i}" not in self._metrics:
                self._metrics[f"BLEU-{i}"] = []
            self._metrics[f"BLEU-{i}"].append(bleu_nltk[i - 1])

        if "sacreBLEU" not in self._metrics:
            self._metrics["sacreBLEU"] = []
        self._metrics["sacreBLEU"].append(bleu_sacre.score / 100)

    # Function to calculate METEOR score
    def _calculate_meteor_scores(self, ref_sentences, hyp_sentences):
        meteor_scores = []
        for ref, hyp in zip(ref_sentences, hyp_sentences):
            meteor = single_meteor_score(
                self.tokenizer(ref).split(), self.tokenizer(hyp).split()
            )
            meteor_scores.append(meteor)
        if "METEOR" not in self._metrics:
            self._metrics["METEOR"] = []
        self._metrics["METEOR"].append(np.mean(meteor_scores))

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
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(np.mean(values))

    # Function to calculate BERT score
    def _calculate_bert_scores(self, ref_sentences, hyp_sentences):
        # Only the F1 score is needed
        bert_p, bert_r, bert_f = self.bert_scorer.score(ref_sentences, hyp_sentences)
        for key, value in zip(
            ["bert_precision", "bert_recall", "bert_f1"], [bert_p, bert_r, bert_f]
        ):
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(np.mean(value.numpy()))

    # Function to calculate CIDEr score
    def _calculate_ciderd_scores(self, ref_sentences, hyp_sentences):
        ref_sentences_as_singletons = [[sentence] for sentence in ref_sentences]
        cider = cider_d(
            hyp_sentences,
            ref_sentences_as_singletons,
            tokenizer=ptbtok.tokenize,
            return_all_scores=False,
        )
        if "CIDEr-D" not in self._metrics:
            self._metrics["CIDEr-D"] = []
        self._metrics["CIDEr-D"].append(cider.item())

    def _calculate_spice_scores(self, ref_sentences, hyp_sentences, java_path=None):
        ref_sentences_as_singletons = [[sentence] for sentence in ref_sentences]
        spice_score = spice(
            hyp_sentences,
            ref_sentences_as_singletons,
            java_path=java_path,
            return_all_scores=False,
        )
        if "SPICE" not in self._metrics:
            self._metrics["SPICE"] = []
        self._metrics["SPICE"].append(spice_score.item())

    def calculate_metrics(
        self,
        ref_sentences: list[str],
        hyp_sentences: list[list[str]],
        out_dir: str | None = None,
        out_name: str | None = None,
        java_path: str | None = None,
    ):
        # hyp_sentences should be a list of list of strings where the dimensions are [n_samples, n_sentences]
        # So you would run predict n_samples amount of times and store the result of predict in a list of size n_sentences
        # ref_sentences should be a list of strings where the dimensions are [n_sentences] and contains the ground truth sentences
        # See the example in main
        metrics_with_conf_int = {}
        for i in tqdm(range(len(hyp_sentences))):
            self._calculate_bleu_scores(ref_sentences, hyp_sentences[i], max_n=4)
            self._calculate_meteor_scores(ref_sentences, hyp_sentences[i])
            self._calculate_rouge_scores(ref_sentences, hyp_sentences[i])
            self._calculate_bert_scores(ref_sentences, hyp_sentences[i])
            self._calculate_ciderd_scores(ref_sentences, hyp_sentences[i])
            # We can't do this without the correct java version
            if java_path:
                self._calculate_spice_scores(
                    ref_sentences, hyp_sentences[i], java_path=java_path
                )

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
            metrics_df.to_csv(os.path.join(out_dir, out_name), index=False)

        return metrics_with_conf_int


if __name__ == "__main__":
    # Example predictions and references
    predictions = [["hello general kenobi, how are you", "foo bar bar bar foobar"]] * 10
    references = [
        "hello there general kenobi, how are you",
        "foo bar bar bar bar foobar",
    ]

    # Initialize the TSCapMetrics class
    ts_cap_metrics = TSCapMetrics()

    print(
        ts_cap_metrics.calculate_metrics(
            references,
            predictions,
            out_dir="./metrics",
            out_name="test.csv",
            java_path="~/jdk-11.0.2/bin/java",
        )
    )
