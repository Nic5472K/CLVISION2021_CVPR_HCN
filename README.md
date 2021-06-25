<pre>
###===###
# CLVISION2021_CVPR_HCN
###===###
This repository is reserved for our CLVISION CVPR2021 workshop paper 
"Plastic and Stable Gated Classifiers for Continual Learning".
(https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Kuo_Plastic_and_Stable_Gated_Classifiers_for_Continual_Learning_CVPRW_2021_paper.pdf)

###===###
We provide 7 scripts in the "Codes" folder.
  Item001 - A001_pMNIST_singular_MLP.py
  Item002 - A002_pMNIST_singular_HCN.py
  Item003 - Az01_Loader.py
  Item004 - Az02_BaseLearner.py
  Item005 - Az03_Eval.py
  Item006 - Az04_UpdateProgress.py
  Item007 - Az22_BaseLearner.py

Item001 uses Item{003, 004, 005, 006} whereas
Item002 uses Item{007, 003, 005, 006}.

These codes are based on our previous work in the repository of
"MetaLearnCC2021_AAAI_MetaSGD-CL".
(https://github.com/Nic5472K/MetaLearnCC2021_AAAI_MetaSGD-CL)
The codes of therein that repository are fully commented, hence
we neglect the duplicated comments therein this repository.

To run the code, just execute Item001 and/or Item002.
Most of the content are identical, and that the only difference is in
the architectural design of the base learner.
Item001 uses the backbone in Item004, which is a multiple layer perceptron;
Item002 uses the backbone in Item007, which is a highway connection network (HCN).

###===###
To cite our work, please use the following bibtex
@InProceedings{Kuo_2021_CVPR,
    author    = {Kuo, Nicholas I-Hsien and Harandi, Mehrtash and Fourrier, Nicolas and Walder, Christian and Ferraro, Gabriela and Suominen, Hanna},
    title     = {Plastic and Stable Gated Classifiers for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {3553-3558}
}

###===###
This repository is currently under construction, please contact NicK via the email address
n.kuo@unsw.edu.au

Date updated: 26th of June, 2021 "eji u.4tp6"
</pre>
