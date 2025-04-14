# A Data-Efficient Nearest-Neighbor Language Model via Lightweight Nets
<div align="center">
<div>
    <a href='' target='_blank'>Qinhao Zhou</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=-D5k5ioAAAAJ&view_op=list_works' target='_blank'>Xiang Xiang</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Ke Wang</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Yuqi Zhang</a><sup>3</sup>&emsp;
    <a href='' target='_blank'>Yuchuan Wu</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Yongbin Li</a><sup>2</sup>&emsp;
</div>
<div>
<sup>1</sup>School of Artificial Intelligence and Automation, Huazhong University of Science and Technology&emsp;

<sup>2</sup>Alibaba Group&emsp;

<sup>3</sup>Nat'l Practice Base for Outstanding Engineers (Digital Tech)&emsp;

</div>
</div>


Code for our CCMT 2024 paper "A Data-Efficient Nearest-Neighbor Language Model via Lightweight Nets". 
Please cite our paper if you find this repository helpful in your research:

```
@inproceedings{zhouxiang-lightweight-ccmt-2024,
    title = "A Data-Efficient Nearest-Neighbor Language Model via Lightweight Nets",
    author = "Qinhao Zhou, Xiang Xiang, Ke Wang, Yuqi Zhang",
    month = Nov,
    year = "2024",
}
```

This project is based on adaptive kNN-MT, 
The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [knn-lm](https://github.com/urvashik/knnlm).


## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

## Run the Code

In line with other works based on kNN-MT, our code is designed to support the following four datasets by default:

| IT      | Medical | koran  | Law      |
|---------|---------|--------|----------|
| 3613350 | 6903320 | 524400 | 19070000 |

The data can be downloaded in [this site](https://github.com/roeeaharoni/unsupervised-domain-clusters)
Pre-trained model ckpt from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md) need to be download before run the code.


#### Train
First, construct the datastore

```
bash ./sh/create_datastore.sh
```

Then, use faiss build datastore index, This step significantly improves the training speed.

```
bash ./sh/build_faiss_index.sh
```

Fianlly, run the train script.

```
bash ./sh/train.sh  
```


#### Test

For inference, we can run 
```
bash ./sh/inference.sh  
```


