# Generalizing to Unseen Domains in Diabetic Retinopathy Classification
**Chamuditha Jayanga Galappaththige**, **Gayal Kuruppu** and **Muhammad Haris Khan**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-FFF933)](https://arxiv.org/abs/2310.17255) 

> **Abstract:** *Diabetic retinopathy (DR) is caused by long-standing diabetes and is among the fifth leading cause for visual impairment. The prospects of early diagnosis and treatment could be helpful in curing the disease, however, the detection procedure is rather challenging and mostly tedious. Therefore, automated diabetic retinopathy classification using deep learning techniques has gained interest in the medical imaging community. Akin to several other real-world applications of deep learning, the typical assumption of i.i.d data is also violated in DR classification that relies on deep learning. Therefore, developing DR classification methods robust to unseen distributions is of great value. In this paper, we study the problem of generalizing a model to unseen distributions or domains (a.k.a domain generalization) in DR classification. To this end, we propose a simple and effective domain generalization (DG) approach that achieves self-distillation in vision transformers (ViT) via a novel prediction softening mechanism. This prediction softening is an adaptive convex combination of one-hot labels with the model’s own knowledge. We perform extensive experiments on challenging open-source DR classification datasets under both multi-source and more challenging single-source DG settings with three different ViT backbones to establish the efficacy and applicability of our approach against competing methods. For the first time, we report the performance of several state-of-the-art domain generalization (DG) methods on open-source DR classification datasets after conducting thorough experiments. Finally, our method is also capable of delivering improved calibration performance than other methods, showing its suitability for safety-critical applications, including healthcare. We hope that our contributions would instigate more DG research across the medical imaging community.*

### Multi-source Domain Generaliztion performances
<p align="center">
     <img src="https://github.com/Chumsy0725/SPSD-ViT/blob/main/Resources/results.png" > 
</p>

We report an extensive comparison with the existing SOTA methods in DG literature on DR datasets as shown in Table 1. We believe that our experiments will offer insights into how the existing SOTA DG methods on natural datasets behave on DR datasets. Moreover, we compare our proposed method with existing SOTA methods in the DR context. We report ERM results with both CNN and ViT backbones as a baseline as it shows competitive performance against many existing DG methods. Therefore, we include ERM method with both CNN and ViT backbones as a baseline. We achieve a notable *+2.1%* increase in (overall) average accuracy over the second-best contestant. 

### Installation and Set-up

This repository is built on [Domainbed](https://github.com/facebookresearch/DomainBed/tree/main) and [SDViT](https://github.com/maryam089/SDViT/tree/main). Please refer aforementioned repositories for installation. <br /> 
Download the dataset to the `./data` directory. You can download the dataset using this [link](https://drive.google.com/file/d/1PX03XTn7mRDE9KNvBfvCiSxlemXaz1-b/view?usp=sharing).<br /> 
Download the necessary checkpoints to the `doaminbed/pretrained_models` directory from [DeiT](https://github.com/facebookresearch/deit), [CvT](https://github.com/microsoft/CvT) and [T2T](https://github.com/yitu-opensource/T2T-ViT). <br /> 
Use `bash run.sh` to launch a training sweep on our best-performing model with the CvT-13 backbone. The checkpoints for trained SPSD-ViT with CvT-13 backbone are available [here](https://drive.google.com/file/d/1pFQavXUFkeKQB8ew-y3KZXjY_Z6oX7g5/view?usp=sharing).

## Citation

```bibtex
@InProceedings{Jayanga_SPSDViT,
    author    = {Galappaththige, Chamuditha Jayanga and Kuruppu, Gayal and Khan, Muhammad Haris},
    title     = {Generalizing to Unseen Domains in Diabetic Retinopathy Classification},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2024}
}
```
