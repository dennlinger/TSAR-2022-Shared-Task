# TSAR-2022-Shared-Task Datasets and Evaluation Scripts
TSAR2022 Shared Task on Lexical Simplification for English, Spanish and Portuguese - Datasets and Evaluation scripts

## Datasets

There is no training dataset for the TSAR-2022 Shared Task. 
However, a sample of 10 or 12 instances with gold standard annotations will be provided as a trial dataset.
<br/> 
<br/> 
Format of the files:
- Format of the *trial_none* and *test_none* files: <span style="font-weight:normal">Sentence&lt;TAB&gt;ComplexWord</span>
- Format of the *trial_gold* and *test_gold* files: <span style="font-weight:normal">Sentence&lt;TAB&gt;ComplexWord&lt;TAB&gt;Annotation1&lt;TAB&gt;Annotation2&lt;TAB&gt;...&lt;TAB&gt;AnnotationN</span>


### Trial dataset
The trial dataset consists of a set of 10 instances (for English and Portuguese) and 12 instances (for English) of a sentence, a target complex word.
The *trial_none* files contain only the instances and the *trial_gold* files contain the instances and set of gold annotations.

- /datasets/trial/tsar2022_en_trial_none.tsv
- /datasets/trial/tsar2022_en_trial_gold.tsv
- /datasets/trial/tsar2022_es_trial_none.tsv
- /datasets/trial/tsar2022_es_trial_gold.tsv
- /datasets/trial/tsar2022_pt_trial_none.tsv
- /datasets/trial/tsar2022_pt_trial_gold.tsv

<br/>



### Test dataset 
On 8th September we will release the test files (*test_none*) (with 369/376 instances) used for the evaluation benchmark:
The *test_none* files contain the instances with the sentences and target complex words.

On 30th September (or sooner) we will release the results of the evaluation benchmark and the *test_gold* files (with 369/376 examples).
The *test_gold* files contain the sentences, target complex words, and gold annotations

## Evaluation Scripts 
(Avalaible on 22th July)


## License

The python scripts follow [AGPL 3.0v license](LICENSE).
The datasets (under the /datasets directory) are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License [CC-BY-NC-SA-4.0](CC-BY-NC-SA-4.0).

