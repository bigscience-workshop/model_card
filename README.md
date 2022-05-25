---
license: bigscience-bloom-rail-1.0
language:
- ak
- ar
- as
- bm
- bn
- ca
- code
- en
- es
- eu
- fon
- fr
- gu
- hi
- id
- ig
- ki
- kn
- lg
- ln
- ml
- mr
- ne
- nso
- ny
- or
- pa
- pt
- rn
- rw
- sn
- st
- sw
- ta
- te
- tn
- ts
- tum
- tw
- ur
- vi
- wo
- xh
- yo
- zh
- zhs
- zht
- zu
---

# <p>BLOOM LM<br/> _BigScience Large Open-source Open-access Multilingual Language Model_ <br/>Model Card</p>
<img src="https://assets.website-files.com/6139f3cdcbbff3a68486761d/613cd8997b270da063e230c5_Tekengebied%201-p-500.png" alt="BigScience Logo" width="200"/>


Version 1.0 / 25.May.2022

## Table of Contents
1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Training Data](#training-data)
4. [Risks and Limitations](#risks-and-limitations)
5. [Evaluation](#evaluation)
6. [Recommendations](#recommendations)
7. [Glossary and Calculations](#glossary-and-calculations)
8. [More Information](#more-information)
9. [Model Card Authors](#model-card-authors)

## Model Details  

### Basics
*This section provides information for anyone who wants to know about the model.*

<details>
<summary>Click to expand</summary> <br/>
    
**Developed by:** BigScience ([website](https://bigscience.huggingface.co))

* All collaborators are either volunteers or have an agreement with their employer. *(Further breakdown of participants forthcoming.)*
    
**Model Type:** Transformer-based Language Model

**Version:** 1.0.0

**Languages:** Multiple; see [training data](#training-data)

**License:** RAIL License v1.0 ([link](https://huggingface.co/spaces/bigscience/license))

**Release Date Estimate:** Monday, 11.July.2022

**Send Questions to:** bigscience-contact@googlegroups.com

**Cite as:** BigScience, _BigScience Language Open-source Open-access Multilingual (BLOOM) Language Model_. International, May 2021-May 2022

**Funded by:** 
    
* The French government.

* Hugging Face ([website](https://huggingface.co)).

* Organizations of contributors.  *(Further breakdown of organizations forthcoming.)*

</details>

### Technical Specifications
*This section provides information for people who work on model development.*

<details>
<summary>Click to expand</summary><br/>

Please see [the BLOOM training README](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#readme) for full details on replicating training.

**Model Architecture:** Modified from Megatron-LM GPT2 (see [paper](https://arxiv.org/abs/1909.08053), [BLOOM Megatron code](https://github.com/bigscience-workshop/Megatron-DeepSpeed)):

* Decoder-only architecture

* Layer normalization applied to word embeddings layer (`StableEmbedding`; see [code](https://github.com/facebookresearch/bitsandbytes), [paper](https://arxiv.org/pdf/2110.02861.pdf))

* ALiBI positional encodings (see [paper](https://arxiv.org/pdf/2108.12409.pdf)), with GeLU activation functions

* 176 billion parameters:

    * 70 layers, 112 attention heads

    * Hidden layers are 14336-dimensional

    * Sequence length of 2048 tokens used (see [BLOOM tokenizer](https://huggingface.co/bigscience/tokenizer), [tokenizer description](#tokenization))

**Objective Function:** Cross Entropy with mean reduction (see [API documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)).
    
**Compute infrastructure:** Jean Zay Public Supercomputer, provided by the French government (see [announcement](https://www.enseignementsup-recherche.gouv.fr/fr/signature-du-marche-d-acquisition-de-l-un-des-supercalculateurs-les-plus-puissants-d-europe-46733)).

* Hardware: 384 A100 80GB GPUs (48 nodes):
    
    * Additional 32 A100 80GB GPUs (4 nodes) in reserve

    *  8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links

    *   CPU: AMD

    *   CPU memory: 512GB per node

    *   GPU memory: 640GB per node

    *   Inter-node connect: Omni-Path Architecture (OPA)

    *   NCCL-communications network: a fully dedicated subnet

    *   Disc IO network: shared network with other types of nodes

* Software:
  
    *   Megatron-DeepSpeed ([Github link](https://github.com/bigscience-workshop/Megatron-DeepSpeed))

    *   DeepSpeed ([Github link](https://github.com/microsoft/DeepSpeed))

    *   PyTorch (pytorch-1.11 w/ CUDA-11.5; see [Github link](https://github.com/pytorch/pytorch))

    *   apex ([Github link](https://github.com/NVIDIA/apex))


#### **Training**

 
_In progress._

- Checkpoint size:
    
    - Bf16 weights: 329GB
    
    - Full checkpoint with optimizer states: 2.3TB

- Training throughput: About 150 TFLOP per GPU per second

- Number of epochs: 1 (*current target*)

- Dates:
    
    - Started 11th March, 2022 11:42am PST

    - Estimated end: 5th July, 2022

- Estimated cost of training: Equivalent of $7-15M

- Server training location: Île-de-France, France

#### **Tokenization**
    
The BLOOM tokenizer ([link](https://huggingface.co/bigscience/tokenizer)) is a learned subword tokenizer trained using:
    
- A byte-level Byte Pair Encoding (BPE) algorithm 

- A simple pre-tokenization rule, no normalization

- A vocabulary size of 250,680

It was trained on a subset of a preliminary version of the corpus using alpha-weighting per language.    
    
</details>


### Environmental Impact

<details>
<summary>Click to expand</summary><br/>

The training supercomputer, Jean Zay ([website](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html)), uses mostly nuclear energy. The heat generated by it is reused for heating campus housing.
    
**Estimated carbon emissions:**  *(Forthcoming upon completion of training.)*
    
**Estimated electricity usage:** *(Forthcoming upon completion of training.)*


</details>
<p>&nbsp;</p>

## Uses

*This section addresses questions around how the model is intended to be used, discusses the foreseeable users of the model (including those affected by the model), and describes uses that are considered out of scope or misuse of the model. 
It provides information for anyone considering using the model or who is affected by the model.*


<details>
<summary>Click to expand</summary><br/>
    
### Intended Use

This model is being created in order to enable public research on large language models (LLMs). LLMs are intended to be used for language generation or as a pretrained base model that can be further fine-tuned for specific tasks. Use cases below are not exhaustive.

#### **Direct Use**

-   Text generation

-   Exploring characteristics of language generated by a language model

    -   Examples: Cloze tests, counterfactuals, generations with reframings

#### **Downstream Use**

-   Tasks that leverage language models include: Information Extraction, Question Answering, Summarization

### Misuse and Out-of-scope Use
*This section addresses what users ought not do with the model.*

See the [BLOOM License](https://huggingface.co/spaces/bigscience/license), Attachment A, for detailed usage restrictions. The below list is non-exhaustive, but lists some easily foreseeable problematic use cases.

#### **Out-of-scope Uses**

Using the model in [high-stakes](#high-stakes) settings is out of scope for this model.  The model is not designed for [critical decisions](#critical-decisions) nor uses with any material consequences on an individual's livelihood or wellbeing. The model outputs content that appears factual but is not correct.  

##### Out-of-scope Uses Include:

-   Usage in biomedical domains, political and legal domains, or finance domains

-   Usage for evaluating or scoring individuals, such as for employment, education, or credit

-   Applying the model for critical automatic decisions, generating factual content, creating reliable summaries, or generating predictions that must be correct

#### **Misuse**

Intentionally using the model for harm, violating [human rights](#human-rights), or other kinds of malicious activities, is a misuse of this model. This includes:

-   Spam generation

-   Disinformation and influence operations

-   Disparagement and defamation

-   Harassment and abuse
  
-   [Deception](#deception)

-   Unconsented impersonation and imitation

-   Unconsented surveillance 

-   Generating content without attribution to the model, as specified in the [RAIL License, Use Restrictions](https://huggingface.co/spaces/bigscience/license)

### Intended Users

#### **Direct Users**

-   General Public

-   Researchers

-   Students

-   Educators

-   Engineers/developers

-   Non-commercial entities

-   Community advocates, including human and civil rights groups

#### Indirect Users

-   Users of derivatives created by Direct Users, such as those using software with an [intended use](#intended-use)

-   Users of [Derivatives of the Model, as described in the License](https://huggingface.co/spaces/bigscience/license)

#### Others Affected (Parties Prenantes)

-   People and groups referred to by the LLM

-   People and groups exposed to outputs of, or decisions based on, the LLM

-   People and groups whose original work is included in the LLM
    
</details>
<p>&nbsp;</p>

## Training Data
*This section provides a high-level overview of the training data. It is relevant for anyone who wants to know the basics of what the model is learning.*


<details>
<summary>Click to expand</summary><br/>
    
Details for each dataset are provided in individual [Data Cards](https://huggingface.co/spaces/bigscience/BigScienceCorpus).

Training data includes:

-   45 natural languages
    
-   12 programming languages

-   In 1.5TB of pre-processed text, converted into 350B unique tokens (see [the tokenizer section](#tokenization) for more.)


#### **Languages**
    
The pie chart shows the distribution of languages in training data.
   
![pie chart showing the distribution of languages in training data](https://github.com/bigscience-workshop/model_card/blob/main/assets/data/pie_chart.svg?raw=true)


The following table shows the further distribution of Niger-Congo and Indic languages in the training data.
<details>
<summary>Click to expand</summary><br/>
    
| Niger Congo    | Percentage |         | Indic     | Percentage |
|----------------|------------ |------  |-----------|------------|
| Chi Tumbuka    | 0.00002    |         | Assamese  | 0.01       |
| Kikuyu         | 0.00004    |         | Odia      | 0.04       |
| Bambara        | 0.00004    |         | Gujarati  | 0.04       |
| Akan           | 0.00007    |         | Marathi   | 0.05       |
| Xitsonga       | 0.00007    |         | Punjabi   | 0.05       |
| Sesotho        | 0.00007    |         | Kannada   | 0.06       |
| Chi Chewa      | 0.0001     |         | Nepali    | 0.07       |
| Setswana       | 0.0002     |         | Telugu    | 0.09       |
| Northern Sotho | 0.0002     |         | Malayalam | 0.10       |
| Fon            | 0.0002     |         | Urdu      | 0.10       |
| Kirundi        | 0.0003     |         | Tamil     | 0.20       |
| Wolof          | 0.0004     |         | Bengali   | 0.50       |
| Kuganda        | 0.0004     |         | Hindi     | 0.70       |
| Chi Shona      | 0.001      |
| Isi Zulu       | 0.001      |
| Igbo           | 0.001      |
| Xhosa          | 0.001      |
| Kinyarwanda    | 0.003      |
| Yoruba         | 0.006      |
| Swahili        | 0.02       |
</details>

The following table shows the distribution of programming languages.
<details>
<summary>Click to expand</summary><br/>
    
| Extension      | Language   | Number of files |
|----------------|------------|-----------------|
| java           | Java       | 5,407,724       |
| php            | PHP        | 4,942,186       |
| cpp            | C++        | 2,503,930       |
| py             | Python     | 2,435,072       |
| js             | JavaScript | 1,905,518       |
| cs             | C#         | 1,577,347       |
| rb             | Ruby       | 6,78,413        |
| cc             | C++        | 443,054         |
| hpp            | C++        | 391,048         |
| lua            | Lua        | 352,317         |
| go             | GO         | 227,763         |
| ts             | TypeScript | 195,254         |
| C              | C          | 134,537         |
| scala          | Scala      | 92,052          |
| hh             | C++        | 67,161          |
| H              | C++        | 55,899          |
| tsx            | TypeScript | 33,107          |
| rs             | Rust       | 29,693          |
| phpt           | PHP        | 9,702           |
| c++            | C++        | 1,342           |
| h++            | C++        | 791             |
| php3           | PHP        | 540             |
| phps           | PHP        | 270             |
| php5           | PHP        | 166             |
| php4           | PHP        | 29              |
    
</details>    
</details>
<p>&nbsp;</p>

## Risks and Limitations
*This section identifies foreseeable harms and misunderstandings.*

<details>
<summary>Click to expand</summary><br/>
    
Model may:

-   Overrepresent some viewpoints and underrepresent others

-   Contain stereotypes
  
-   Contain [personal information](#personal-data-and-information)

-   Generate:

    -   Hateful, abusive, or violent language

    -   Discriminatory or prejudicial language

    -   Content that may not be appropriate for all settings, including sexual content

-   Make errors, including producing incorrect information as if it were factual

-   Generate irrelevant or repetitive outputs
</details>
<p>&nbsp;</p>

## Evaluation
*This section describes the evaluation protocols and provides the results.*

<details>
<summary>Click to expand</summary><br/>

### Metrics 
*This section describes the different ways performance is calculated and why.*
    
Includes:

| Metric             | Why chosen                                                         |
|--------------------|--------------------------------------------------------------------|
| [Perplexity](#perplexity)         | Standard metric for quantifying model improvements during training |
| Cross Entropy [Loss](#loss) | Standard objective for language models                             |

And multiple different metrics for specific tasks. _(More evaluation metrics forthcoming upon completion of evaluation protocol.)_

### Factors 
*This section lists some different aspects of what BLOOM models. Its focus is on those aspects that are likely to give rise to high variance in model behavior.*

- Language, such as English or Yoruba

- Domain, such as newswire or stories
    
- Demographic characteristics, such as gender or nationality

###  Results
*Results are based on the [Factors](#factors) and [Metrics](#metrics).*

**Train-time Evaluation:**

As of 19.May.2022, 18:00:

- Training Loss: 2.04

- Validation Loss: 2.21

- Perplexity: 9.15

(More evaluation scores forthcoming at the end of model training.)

</details>
<p>&nbsp;</p>

## Recommendations

*This section provides information on warnings and potential mitigations.*


<details>
<summary>Click to expand</summary><br/>

-   Indirect users should be made aware when the content they're working with is created by the LLM.

-   Users should be aware of [Risks and Limitations](#risks-and-limitations), and include an appropriate age disclaimer or blocking interface as necessary.

-   Models pretrained with the LLM should include an updated Model Card.

-   Users of the model should provide mechanisms for those affected to provide feedback, such as an email address for comments.

</details>
<p>&nbsp;</p>

## Glossary and Calculations

*This section defines common terms and how metrics are calculated.*



<details>
<summary>Click to expand</summary><br/>

-   <a name="loss">**Loss:**</a> A calculation of the difference between what the model has learned and what the data shows ("groundtruth"). The lower the loss, the better. The training process aims to minimize the loss. 

-   <a name="perplexity">**Perplexity:**</a> This is based on what the model estimates the probability of new data is. The lower the perplexity, the better.  If the model is 100% correct at predicting the next token it will see, then the perplexity is 1. Mathematically this is calculated using entropy. 

-   <a name="high-stakes">**High-stakes settings:**</a> Such as those identified as "high-risk AI systems" and "unacceptable risk AI systems" in the European Union's proposed [Artificial Intelligence (AI) Act](https://artificialintelligenceact.eu/annexes/).

-   <a name="critical-decisions">**Critical decisions:**</a> Such as those defined in [the United States' proposed Algorithmic Accountability Act](https://www.congress.gov/117/bills/s3572/BILLS-117s3572is.pdf).

-   <a name="human-rights">**Human rights:**</a> Includes those rights defined in the [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf).

-  <a name="personal-data-and-information">**Personal Data and Personal Information:**</a> Personal data and information is defined in multiple data protection regulations, such as "[personal data](https://gdpr-info.eu/issues/personal-data/)" in the [European Union's General Data Protection Regulation](https://gdpr-info.eu); and "personal information" in the Republic of South Africa's [Protection of Personal Information Act](https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013popi.pdf), The People's Republic of China's [Personal information protection law](http://en.npc.gov.cn.cdurl.cn/2021-12/29/c_694559.htm).
  
- <a name="sensitive-characteristics">**Sensitive characteristics:**</a> This includes specifically protected categories in human rights (see [UHDR, Article 2](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf)) and personal information regulation (see GDPR, [Article 9; Protection of Personal Information Act, Chapter 1](https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013popi.pdf))

- <a name="deception">**Deception:**</a> Doing something to intentionally mislead individuals to believe something that is false, such as by creating deadbots or chatbots on social media posing as real people, or generating text documents without making consumers aware that the text is machine generated.

</details>
<p>&nbsp;</p>

## More Information

<details>
<summary>Click to expand</summary><br/>
    
### Dataset Creation

Blog post detailing the design choices during the dataset creation: https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling

### Technical Specifications

Blog post summarizing how the architecture, size, shape, and pre-training duration where selected: https://bigscience.huggingface.co/blog/what-language-model-to-train-if-you-have-two-million-gpu-hours

More details on the architecture/optimizer: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Blog post on the hardware/engineering side: https://bigscience.huggingface.co/blog/which-hardware-to-train-a-176b-parameters-model

Details on the distributed setup used for the training: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Tensorboard updated during the training: https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard#scalars&tagFilter=loss

Insights on how to approach training, negative results: https://github.com/bigscience-workshop/bigscience/blob/master/train/lessons-learned.md

Details on the obstacles overcome during the preparation on the engineering side (instabilities, optimization of training throughput, so many technical tricks and questions): https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md

### Initial Results

Initial prompting experiments using interim checkpoints: https://huggingface.co/spaces/bigscience/bloom-book

</details>
<p>&nbsp;</p>
    
## Model Card Authors
*Ordered roughly chronologically and by amount of time spent.*

Margaret Mitchell, Giada Pistilli, Yacine Jernite, Ezinwanne Ozoani, Marissa Gerchick, Nazneen Rajani, Sasha Luccioni, Irene Solaiman, Maraim Masoud, Somaieh Nikpoor, Carlos Muñoz Ferrandis, Stas Bekman, Danish Contractor, David Lansky, Angelina McMillan-Major, Tristan Thrush, Christopher Akiki, Suzana Ilić, Gérard Dupont, Shayne Longpre, Manan Dey, Stella Biderman, Douwe Kiela, Emi Baylor, Teven Le Scao, Aaron Gokaslan, Julien Launay
