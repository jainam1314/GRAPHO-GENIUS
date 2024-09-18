<p align="center">
    <h1 align="center">GRAPHO-GENIUS</h1>
</p>
<p align="center">
    <em><code>❯ Machine Learning based Personality Trait Analysis Tool</code></em>
</p>
<p align="center">
    <!-- <img src="https://img.shields.io/github/license/jainam1314/GRAPHO-GENIUS?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license"> -->
    <img src="https://img.shields.io/github/last-commit/jainam1314/GRAPHO-GENIUS?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/jainam1314/GRAPHO-GENIUS?style=flat&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/jainam1314/GRAPHO-GENIUS?style=flat&color=0080ff" alt="repo-language-count">
</p>
<p align="center">
        <em>Built with the tools and technologies:</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/HTML5-E34F26.svg?style=flat&logo=HTML5&logoColor=white" alt="HTML5">
    <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
</p>

<br>

#####  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)

---

##  Overview

<code>❯ GraphoGenius presents a novel deep learning architecture for comprehensive personality 
assessment from handwritten text. The model prioritizes a granular approach, focusing 
on the analysis of individual letters and their corresponding graphological features. This 
aligns with established graphological principles, where specific letter formations are 
believed to hold associations with personality traits. In contrast to existing systems that 
may emphasize global features like writing slant or margins, our model delves into a more 
detailed examination of the written form. This in-depth analysis aims to provide a more 
holistic and nuanced understanding of an individual's personality. The potential outcome 
is a tool for self-discovery and personal growth, empowering users to leverage insights 
from their handwriting analysis to potentially modify their writing habits for self
improvement. By moving beyond a purely entertainment-oriented purpose, 
GraphoGenius aspires to contribute to the field of graphological personality assessment 
through a data-driven and technologically advanced approach.</code>

---

##  Features

<code>❯ Individual letter based trait detection to ensure detailed analysis</code> 
<br>
<code>❯ Multiple report formats to ensure ease of understanding for the user of jargons and terminologies of graphology</code>
<br>
<code>❯ Summarised report allows user to get an overview of the identified traits giving a quick brief</code>
<br>
<code>❯ Detailed report includes individual letter based summary with all the personality traits associated with a group of letters allowing keen users to focus on the traits they desire</code>




---

##  Repository Structure

```sh
└── GRAPHO-GENIUS/
    ├── app.py
    ├── config.json
    ├── models
    │   ├── poly
    │   ├── type_identification
    │   └── type_identification_small
    ├── ocr.py
    ├── ocr_2.py
    ├── ocr_3.py
    ├── sample_1.png
    ├── sample_7.png
    ├── scripts
    │   ├── Extracted_Features
    │   ├── Pre-Processing.py
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── categorize.py
    │   ├── extract.py
    │   ├── extractor.py
    │   ├── feature_routine.py
    │   ├── label_routine.py
    │   ├── routine_extract.py
    │   ├── test.py
    │   └── train.py
    ├── segmentation.py
    ├── static
    │   ├── images
    │   ├── index.js
    │   └── uploads
    ├── templates
    │   ├── analysis_output.html
    │   ├── base.html
    │   ├── index.html
    │   ├── predict.html
    │   └── result.html
    ├── type_identification_A.h5
    ├── type_identification_b.h5
    ├── type_identification_o.h5
    └── type_identification_s.h5
```
---