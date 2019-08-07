# Skeleton Images Representation (SkeleMotion and TSRJI)

This repository holds the skeleton image representation codes for the papers
 
> 
**SkeleMotion: A New Representation of Skeleton Joint Sequences Based on Motion Information for 3D Action Recognition**,
Carlos Caetano, Jessica Sena, François Brémond, Jefersson A. dos Santos, and William Robson Schwartz,
*AVSS 2019*, Taipei, Taiwan.
>
[[Arxiv Preprint](https://arxiv.org/abs/1907.13025)]

> 
**Skeleton Image Representation for 3D Action Recognition based on Tree Structure and Reference Joints**,
Carlos Caetano, François Brémond and William Robson Schwartz,
*SIBGRAPI 2019*, Rio de Janeiro, Brazil.
>


# Contents
* [Usage Guide](#usage-guide)
  * [Prerequisites](#prerequisites)
  * [Code & Data Preparation](#code--data-preparation)
    * [Get the code](#get-the-code)
    * [Get the Depth Maps](#get-the-depth-maps)
* [Other Info](#other-info)
  * [Citation](#citation)
  * [Contact](#contact)

----
# Usage Guide

## Prerequisites
[[back to top](#skeleton-images-representation-SkeleMotion-and-SRJI)]

The main dependencies to run the code are

- [OpenCV][opencv]
- [NumPy][numpy]

The codebase is written in Python 3.6. We recommend the [Anaconda][anaconda] Python distribution.

## Code & Data Preparation

### Get the code
[[back to top](#skeleton-images-representation-SkeleMotion-and-SRJI)]

Use git to clone this repository
```
git clone --recursive https://github.com/carloscaetano/skeleton-images
```

### Get the Depth Maps
[[back to top](#skeleton-images-representation-SkeleMotion-and-SRJI)]

We experimented our skeleton images representation on two large-scale 3D action recognition datasets: [NTURGB+D 60][nturgb-d60] and [NTURGB+D 120][nturgb-d120]. For more information about accessing the "NTU RGB+D" and "NTU RGB+D 120" datasets, go to [ROSE website][rose].

### Usage
[[back to top](#skeleton-images-representation-SkeleMotion-and-SRJI)]

TODO...

# Other Info
[[back to top](#skeleton-images-representation-SkeleMotion-and-SRJI))]

## Citation
Please cite the following papers if you feel this repository useful.
```
@inproceedings{Caetano:AVSS:2019,
  author    = {Carlos Caetano and
               Jessica Sena and
               François Brémond and
               Jefersson A. dos Santos and
               William Robson Schwartz},
  title     = {SkeleMotion: A New Representation of Skeleton Joint Sequences Based on Motion Information for 3D Action Recognition},
  booktitle   = {IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS)},
  year      = {2019},
}

@inproceedings{Caetano:SIBGRAPI:2019,
  author    = {Carlos Caetano and
               François Brémond and
               William Robson Schwartz},
  title     = {Skeleton Image Representation for 3D Action Recognition based on Tree Structure and Reference Joints},
  booktitle   = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year      = {2019},
}
```

## Contact
For any question, please contact
```
Carlos Caetano: carlos.caetano@dcc.ufmg.br
```

[nturgb-d60]:https://github.com/shahroudy/NTURGB-D
[nturgb-d120]:https://github.com/shahroudy/NTURGB-D
[rose]:http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
[anaconda]:https://www.continuum.io/downloads
[opencv]:https://opencv.org/
[numpy]:https://numpy.org/
