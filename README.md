# MedICaT
MedICaT is a dataset of medical images, captions, subfigure-subcaption annotations, and inline textual references. Instructions for access are provided here.

Figures and captions are extracted from open access articles in PubMed Central and corresponding reference text is derived from [S2ORC](https://github.com/allenai/s2orc).

The dataset consists of:
* 217,060 figures from 131,410 open access papers
* 7507 subcaption and subfigure annotations for 2069 compound figures
* Inline references for ~25K figures in the [ROCO dataset](https://github.com/razorx89/roco-dataset)

A sample of the data is available in `sample/`.

An example data entry:

```
{
  "pdf_hash": "57c9ad0f4aab133f96d40992c46926fabc901ffa",
  "fig_key": "Figure1",
  "fig_uri": "2-Figure1-1.png",
  "s2_caption": "Figure 1. (A) Barium enema and (B) endoscopic image of the high-grade distal colonic obstruction caused by a 5-cm anastomotic stricture.",
  "s2orc_caption": "Figure 1. (A) Barium enema and (B) endoscopic image of the high-grade distal colonic obstruction caused by a 5-cm anastomotic stricture.",
  "s2orc_references": [
    "Computed tomography (CT) showed a distal large bowel obstruction, and a barium enema revealed a high-grade stenosis proximal to the anastomotic site in the recto-sigmoid region (Figure 1 ).",
    "Flexible sigmoidoscopy revealed a tight, fibrotic, benign-appearing anastomotic stricture 15 cm from the anal verge ( Figure 1) ."
  ],
  "radiology": false,
  "scope": true,
  "predicted_type": "Medical images",
  "oa_info": {
    "doi": "10.14309/crj.2014.54",
    "doi_url": "https://doi.org/10.14309/crj.2014.54",
    "oa": {
      "is_oa": true,
      "oa_status": "gold",
      "journal_is_oa": true,
      "journal_is_in_doaj": true,
      "license": "cc-by-nc-nd",
      "provenance": "unpaywall"
    }
  }
}
```

The corresponding figure is located at `figures/57c9ad0f4aab133f96d40992c46926fabc901ffa_2-Figure1-1.png` (`{pdf_hash}_{fig_uri}`).

### To download:

Please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSdB6w2HHNtD-v6SJr3wFMQl8WxR-wigrfVJPvqI-RR50miI7w/viewform) for access. We will respond to your request within 48 hours.

### Code
Please see the `code` directory for the code associated with our paper. The `code/README.md` includes additional information about how you can use this code.

### To cite:

If using this dataset, please cite:

```
@inproceedings{subramanian-2020-medicat,
    title={{MedICaT: A Dataset of Medical Images, Captions, and Textual References}},
    author={Sanjay Subramanian, Lucy Lu Wang, Sachin Mehta, Ben Bogin, Madeleine van Zuylen, Sravanthi Parasa, Sameer Singh, Matt Gardner, and Hannaneh Hajishirzi},
    year={2020},
    booktitle={Findings of EMNLP},
}
```

### License

Each source document in MedICaT is licensed differently. Articles included in MedICaT have open access licenses (see [CC](https://creativecommons.org/licenses/) and [UPW](https://support.unpaywall.org/support/solutions/folders/44000384007)) or are in the public domain. The license for each article is provided in the associated entry in the dataset. Please abide by these licenses when using. The MedICaT dataset is available for non-commercial use only.

## Contact us

**Email:** `{sanjays, lucyw}@allenai.org`

