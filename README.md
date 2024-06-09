# Document-level Claim Extraction and Decontextualisation for Fact-Checking
A code implementation of this paper "Document-level Claim Extraction and Decontextualisation for Fact-Checking" (<a href="https://arxiv.org/pdf/2406.03239">ACL 2024</a>). 


## Data

Download raw datas from <a href="https://github.com/MichSchli/AVeriTeC">AVeriTeC</a>.


## Components
All models we rely on are pre-trained models (e.g., BertSum) and approaches that do not require training (e.g., BM25).

* Step 1. extracts URLs (the URL linking to the original
web article of the claim) available for claim extraction and corresponding text data from AVeriTeC. 
    ```bash
    python 1_extract_texts_from_url.py
    ``` 
* Step 2. generates high-quality context for decontextualisation. 
  * __Sentence Ranking__: download from <a href="https://github.com/nlpyang/PreSumm">here</a>.
  * __Text Entailment__: download from <a href="https://github.com/khuangaf/ZeroFEC?tab=readme-ov-file">here</a>
  * __Candidate Answer Extraction__: download spacy (python -m spacy download en_core_web_lg) to extract ambiguous information units.
  * __Question Generation__: download from <a href="https://huggingface.co/Salesforce/mixqg-base">here</a>.
  * __Question Answering__: download from <a href="https://github.com/allenai/unifiedqa">here</a>.
  * __QA-to-Context__: follow <a href="https://github.com/nlpyang/PreSumm">this repository</a> to download the QA2D model from <a href="https://console.cloud.google.com/storage/browser/few-shot-fact-verification;tab=objects?authuser=0&prefix=&forceOnObjectsSortingFiltering=false">here</a>.
  * __High-quality Context Generation__: download from <a href="https://github.com/nlpyang/PreSumm">here</a>.
    ```bash
    python 2_context_generation.py 
    ```
* Step 3. decontextualises candidate central sentences with generated qa pairs. (Download the decontextualsation model from <a href="https://github.com/google-research/language/tree/master/language/decontext">here</a>):)
    ```bash
    python 3_decontextualisation.py 
    ``` 


  




## Citation

If you use this code useful, please star our repo or consider citing:
```
@misc{deng2024documentlevel,
      title={Document-level Claim Extraction and Decontextualisation for Fact-Checking}, 
      author={Zhenyun Deng and Michael Schlichtkrul and Andreas Vlachos},
      year={2024},
      eprint={2406.03239},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
