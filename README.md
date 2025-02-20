# awesome-fashion-ai

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> [ayushidalmia/awesome-fashion-ai: 474ef4c](https://github.com/ayushidalmia/awesome-fashion-ai)

精心整理的清单，涵盖与时尚和电子商务领域人工智能相关的研究论文、数据集、工具、会议及研讨会。

## 目录

* [论文](#papers)
* [研讨会](#workshops)
* [教程](#tutorials)
* [数据集](#datasets)
* [五花八门](#miscellaneous)

### 论文<a name='papers'></a>

领域

* [时尚嵌入](#fashion-embeddings)
* [个性化/推荐/服装搭配/兼容性](#personalisationrecommendationoutfit-compositioncompatibility)
* [视觉搜索/视觉推荐/视觉检索](#visual-searchvisual-recommendationvisual-retrieval)
* [时尚领域中的自然语言理解](#natural-language-understanding-in-fashion)
* [时尚图像目标检测/分类/解析/分割/属性操控](#fashion-image-object-detectionclassificationparsingsegmentationattribute-manipulation)
* [零售洞察/趋势/预测/库存管理](#retail-insightstrendsforecastinginventory-management)
* [图像生成/时尚领域的图像操控/风格迁移](#image-generationimage-manipulation-in-fashionstyle-transfer)
* [造型/场合](#stylingoccasion)
* [社交媒体](#social-media)
* [尺码选择/虚拟试衣间](#sizingvirtual-trial-room)
* [视频](#video)
* [多模态](#multimodal)
* [对话/交流](#dialog-conversation)
* [服装模特](#clothing-model)
* [从图像生成 3D 服装](#3d-clothing-from-images)

##### 时尚嵌入<a name='fashion-embeddings'></a>

- [用于时尚兼容性的半监督视觉表征学习](https://arxiv.org/abs/2109.08052)  ACM RecSys 2021
- [128 维浮点表示时尚风格：利用弱数据进行联合排序与分类以提取特征](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Simo-Serra_Fashion_Style_in_CVPR_2016_paper.pdf)
- [学习类型感知嵌入用于时尚兼容性](https://arxiv.org/pdf/1803.09196v1.pdf), ECCV, 2018
- [Style2Vec：从风格集学习时尚单品的表征](https://arxiv.org/pdf/1708.04014v1.pdf)
- [上下文感知的视觉兼容性预测](https://arxiv.org/abs/1902.03646)
  
##### 个性化/推荐/服装搭配/兼容性<a name='#personalisationrecommendationoutfit-compositioncompatibility'></a>

- [嗨，神奇衣橱，告诉我穿什么！](https://people.cs.clemson.edu/~jzwang/1501863/mm2012/p619-liu.pdf), MM, 2012
- [时尚正在成型：基于网络资源中体型理解服装偏好](https://arxiv.org/pdf/1807.03235v1.pdf)
- [从时尚图像创建胶囊衣橱](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hsiao_Creating_Capsule_Wardrobes_CVPR_2018_paper.pdf)
- [时尚中的神经美学：时尚感的感知建模](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Simo-Serra_Neuroaesthetics_in_Fashion_2015_CVPR_paper.pdf)
- [用于定制时尚服装搭配的可解释分区嵌入](https://arxiv.org/pdf/1806.04845v4.pdf)
- [基于生成图像模型的视觉感知时尚推荐与设计](https://arxiv.org/pdf/1711.02231v1.pdf)
- [基于长短期记忆网络（LSTM）的动态客户模型用于时尚推荐](https://arxiv.org/pdf/1708.07347v1.pdf)
- [面向个性化的产品特征描述：从非结构化数据中学习属性以推荐时尚产品](https://arxiv.org/pdf/1803.07679v1.pdf)
- [基于集合数据的端到端深度学习方法挖掘时尚服装搭配](https://arxiv.org/pdf/1608.03016v2.pdf)
- [利用双向长短期记忆网络学习时尚兼容性](https://arxiv.org/pdf/1707.05691v1.pdf)
- [通过异构二元共现学习视觉服装风格](https://arxiv.org/pdf/1509.07473v1.pdf)
- [从个人衣橱推荐服装搭配](https://arxiv.org/pdf/1804.09979v1.pdf)
- [迈向可解释的时尚推荐](https://arxiv.org/pdf/1901.04870v1.pdf) 
  
##### 视觉搜索/视觉推荐/视觉检索<a name='#visual-searchvisual-recommendationvisual-retrieval'></a>

- [Studio2Shop：从摄影棚拍摄到时尚单品](https://arxiv.org/pdf/1807.00556v1.pdf)
- [通过定位学习属性表征以实现灵活的时尚搜索](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ak_Learning_Attribute_Representations_CVPR_2018_paper.pdf)
- [学习潜在 “造型”：从时尚图像中无监督发现风格一致的嵌入](https://arxiv.org/pdf/1707.03376v2.pdf)
- [自动空间感知时尚概念发现](https://arxiv.org/pdf/1708.01311v1.pdf)
- [利用弱标注数据进行时尚图像检索与标签预测](https://arxiv.org/pdf/1709.09426v1.pdf)
- [基于街拍时尚图像的大规模视觉推荐](https://arxiv.org/pdf/1401.1778v1.pdf)
- [学习统一嵌入用于服装识别](https://arxiv.org/pdf/1707.05929.pdf)
- [基于深度学习的大规模电子商务视觉推荐与搜索](https://arxiv.org/pdf/1703.02344v1.pdf)
- [eBay 的视觉搜索](https://arxiv.org/pdf/1706.03154v2.pdf)
- [基于生成图像模型的视觉感知时尚推荐与设计](https://arxiv.org/pdf/1711.02231v1.pdf)
- [基于双属性感知排序网络的跨域图像检索](https://arxiv.org/pdf/1505.07922v1.pdf)
- [基于图像的风格与替代品推荐](https://arxiv.org/pdf/1506.04757v1.pdf)
- [从 T 台到日常：时尚的视觉分析](http://www.tamaraberg.com/papers/runway_to_realway.pdf)
- [打造造型：日常照片中用于自动产品推荐的服装识别与分割](http://image.ntua.gr/iva/files/kalantidis_icmr13.pdf), ICMR 2013
- [DeepStyle：时尚与室内设计的多模态搜索引擎](https://arxiv.org/abs/1801.03002v2)

##### 时尚领域中的自然语言理解<a name='natural-language-understanding-in-fashion'></a>

- [用于时尚的分层深度学习自然语言解析器](https://arxiv.org/pdf/1806.09511v1.pdf)
- [“让我说服你购买我的产品……”：时尚产品自动化说服系统的案例研究](https://arxiv.org/pdf/1709.08366v1.pdf)
- [“设计个人时尚的未来”](http://ranjithakumar.net/resources/personal_fashion.pdf)
- [用于电子商务产品属性提取的深度循环神经网络](https://arxiv.org/pdf/1803.11284v1.pdf)
  
##### 时尚图像目标检测/分类/解析/分割/属性操控<a name='fashion-image-object-detectionclassificationparsingsegmentationattribute-manipulation'></a>

- [如何从社交媒体中提取时尚趋势？](https://arxiv.org/pdf/1806.10787v1.pdf)
- [用于时尚地标检测与服装类别分类的注意力时尚语法网络](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Attentive_Fashion_Grammar_CVPR_2018_paper.pdf)
- [用于交互式时尚搜索的记忆增强属性操控网络](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Memory-Augmented_Attribute_Manipulation_CVPR_2017_paper.pdf)
- [DeepFashion：利用丰富标注助力强大的服装识别与检索](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)
- [用于时尚图像分类的结构化输出统一模型](https://arxiv.org/pdf/1806.09445v1.pdf)
- [面向个性化的产品特征描](https://arxiv.org/pdf/1803.07679v1.pdf)
- [融合分层卷积特征用于人体分割与服装时尚分类](https://arxiv.org/pdf/1803.03415v2.pdf)
- [如何从社交媒体中提取时尚趋势？一种支持无监督学习的鲁棒目标检测器](https://arxiv.org/pdf/1806.10787v1.pdf)
- [通过分层循环变换器网络进行无约束时尚地标检测](https://arxiv.org/pdf/1708.02044v1.pdf)
- [时尚服装检测：深度卷积神经网络与姿态相关先验的作用](https://arxiv.org/pdf/1411.5319v2.pdf)
- [解析时尚照片中的服装](http://www.tamaraberg.com/papers/parsingclothing.pdf)
- [纸娃娃解析：检索相似风格以解析服装单品](http://www.tamaraberg.com/papers/paperdoll.pdf)
- [用于时尚识别的双流多任务网络](https://arxiv.org/abs/1901.10172v2)
- [用于时尚地标检测的空间感知非局部注意力机制](https://arxiv.org/abs/1903.04104v1)
- [使用特征金字塔网络进行时尚图像的语义分割](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Martinsson_Semantic_Segmentation_of_Fashion_Images_Using_Feature_Pyramid_Networks_ICCVW_2019_paper.html)
  
##### 零售洞察/趋势/预测/库存管理<a name='retail-insightstrendsforecastinginventory-management'></a>

- [FashionBrain 项目：理解欧洲时尚数据世界的愿景](https://arxiv.org/pdf/1710.09788v1.pdf)
- [时尚前沿：预测时尚视觉风格](https://arxiv.org/pdf/1705.06394v2.pdf)
- [当时尚遇上大数据：畅销服装特征的判别式挖掘](https://arxiv.org/pdf/1611.03915v2.pdf)
- [迈向预测时尚图像的受欢迎程度](https://arxiv.org/pdf/1511.05296v2.pdf)
- [谁引领服装时尚：风格、颜色还是纹理？一项计算研究](https://arxiv.org/pdf/1608.07444v1.pdf)
- [时尚产业中稳健的订单调度：一种多目标优化方法](https://arxiv.org/pdf/1702.00159v1.pdf)
- [变化中的时尚文化](https://arxiv.org/pdf/1703.07920v1.pdf)
- [销售潜力：时尚产品视觉美学可销售性建模](https://kddfashion2017.mybluemix.net/final_submissions/ML4Fashion_paper_10.pdf)
- [ARMDN：用于电子零售需求预测的关联与循环混合密度网络](https://arxiv.org/pdf/1803.03800.pdf)
  
##### 图像生成/时尚领域的图像操控/风格迁移<a name='image-generationimage-manipulation-in-fashionstyle-transfer'></a>

- [联合判别与生成学习用于行人重识别](https://arxiv.org/abs/1904.07223), CVPR 2019 [[项目]](http://zdzheng.xyz/DG-Net/) [[论文]](https://arxiv.org/abs/1904.07223) [[YouTube]](https://www.youtube.com/watch?v=ubCrEAIpQs4) [[哔哩哔哩]](https://www.bilibili.com/video/av51439240) [[Poster]](http://zdzheng.xyz/images/DGNet_poster.pdf)
- [条件类比生成对抗网络：在人物图像上交换时尚单品](https://arxiv.org/pdf/1709.04695v1.pdf)
- [基于特征变换的语言引导时尚图像操控](https://arxiv.org/pdf/1808.04000v1.pdf)
- [SwapNet：基于图像的服装转移](http://openaccess.thecvf.com/content_ECCV_2018/papers/Amit_Raj_SwapNet_Garment_Transfer_ECCV_2018_paper.pdf), ECCV 2018
- [兼容且多样的时尚图像修复](https://arxiv.org/abs/1902.01096v1)
- [Fashion++：对服装搭配进行最小化编辑以提升效果](https://arxiv.org/abs/1904.09261)
- [时尚领域语义分割数据的生成建模](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Korneliusson_Generative_Modelling_of_Semantic_Segmentation_Data_in_the_Fashion_Domain_ICCVW_2019_paper.html)

##### 造型/场合<a name='stylingoccasion'></a>

- [潮人之战：发现时尚风格元素](http://www.tamaraberg.com/papers/hipster_eccv14.pdf)
  
##### 社交媒体<a name='social-media'></a>

- [时尚还是社交：在线时尚网络中的视觉流行度分析](http://www.tamaraberg.com/papers/kota_acm14.pdf)
- [识别社交网络中的时尚账号](https://kddfashion2017.mybluemix.net/final_submissions/ML4Fashion_paper_21.pdf)
  
##### 尺码选择/虚拟试衣间<a name='sizingvirtual-trial-room'></a>

- [在度量空间中分解合身语义用于产品尺寸推荐](https://cseweb.ucsd.edu/~m5wan/paper/recsys18_rmisra)
- [M2E-Try On Net：从模特到大众的时尚试穿](https://arxiv.org/pdf/1811.08599v1.pdf)
  
##### 视频<a name='video'></a>

- [像明星一样穿着：从视频中检索时尚产品](https://arxiv.org/pdf/1710.07198v1.pdf)
- [Video2Shop：将视频中的服装与在线购物图像精确匹配](https://arxiv.org/abs/1804.05287v1), CVPR, 2017

##### 多模态<a name='multimodal'></a>

- [DeepStyle：时尚与室内设计的多模态搜索引擎](https://arxiv.org/pdf/1801.03002v1.pdf)
 
##### 对话/交流<a name='dialog-conversation'></a>

- [网民对时尚照片的风格化评论：数据集与多样性度量](https://arxiv.org/pdf/1801.10300v1.pdf)
- [Instagram 上的时尚对话数据](https://arxiv.org/pdf/1704.04137.pdf)
  
##### 服装模特<a name='clothing-model'></a>

- [DeepWrinkles：精确且逼真的服装建模](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zorah_Laehner_DeepWrinkles_Accurate_and_ECCV_2018_paper.pdf), ECCV 2018
- [学习用于多模态服装设计的共享形状空间](https://geometry.cs.ucl.ac.uk/projects/2018/garment_design/), SIGGRAPH Asia 2018
- [Garnet：用于快速准确 3D 服装模拟的双流网络](https://www.epfl.ch/labs/cvlab/research/garment-simulation/garnet/), ICCV 2019
- [基于学习的虚拟试穿服装动画](http://dancasas.github.io/projects/LearningBasedVirtualTryOn/), Eurographics 2019
- [TailorNet：根据人体姿态、体型和服装风格预测 3D 服装](https://virtualhumans.mpi-inf.mpg.de/tailornet/), CVPR 2020
- [SIZER：用于解析 3D 服装并学习尺寸敏感 3D 服装的数据集与模型](https://virtualhumans.mpi-inf.mpg.de/sizer/), ECCV 2020
 
##### 从图像生成 3D 服装<a name='3d-clothing-from-images'></a>

- [多服装网络：从图像中学习为 3D 人物着装](https://virtualhumans.mpi-inf.mpg.de/mgn/), ICCV 2019
- [Deep Fashion3D：用于从单张图像进行 3D 服装重建的数据集与基准](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited/), ECCV 2020 
- [BCNet：从单张图像学习人体与服装形状](https://arxiv.org/abs/2004.00214), ECCV 2020

### 研讨会<a name='workshops'></a>

* 知识发现与数据挖掘会议（KDD） “人工智能助力时尚” 研讨会 [2020](https://kddfashion2020.mybluemix.net/), [2019](https://kddfashion2019.mybluemix.net/), [2018](https://kddfashion2018.mybluemix.net/), [2017](https://kddfashion2017.mybluemix.net/), [2016](http://kddfashion2016.mybluemix.net/)
* 国际计算机视觉大会（ICCV）/ 欧洲计算机视觉大会（ECCV）时尚、艺术与设计领域的计算机视觉研讨会 [2020](https://sites.google.com/view/cvcreative2020), [2019](https://sites.google.com/view/cvcreative) [2018](https://sites.google.com/view/eccvfashion/), [2017](https://sites.google.com/zalando.de/cvf-iccv2017/home?authuser=0)
* 国际计算机学会信息检索大会（SIGIR）电子商务研讨会 [2018](https://sigir-ecom.github.io/index.html), [2017](http://sigir-ecom.weebly.com/)
* 时尚领域中的推荐系统 [2020](https://fashionxrecsys.github.io/fashionxrecsys-2020/),  [2019](https://zalandoresearch.github.io/fashionxrecsys/)

### 教程<a name='tutorials‘></a>

* [从概念到代码：用于时尚推荐的深度学习](https://www2019.thewebconf.org/tutorials).        
主办方: Omprakash Sonie, Muthusamy Chelliah and Shamik Sural, 2019年万维网大会


### 数据集<a name='datasets'></a>

* [Large-scale Fashion (DeepFashion)](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* [Street2Shop](http://www.tamaraberg.com/street2shop/)
* [Fashionista](http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/)
* [Paperdoll](http://vision.is.tohoku.ac.jp/~kyamagu/research/paperdoll/)
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [Fashion Takes Shape](https://www.groundai.com/project/fashion-is-taking-shape-understanding-clothing-preference-based-on-body-shape-from-online-sources/1)
* [ModaNet](https://github.com/eBay/modanet) [paper](https://arxiv.org/pdf/1807.01394v2.pdf)
* [DeepFashion2](https://github.com/switchablenorms/DeepFashion2),[paper](https://arxiv.org/abs/1901.07973)
* [iMaterialist-Fashion](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6)
* [用于尺码推荐的服装合身度数据集](https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation)
* [多服装网络：从图像中学习为 3D 人体着装](https://virtualhumans.mpi-inf.mpg.de/mgn/), ICCV 2019
* [TailorNet：依据人体姿态、体型和服装风格预测 3D 服装](https://virtualhumans.mpi-inf.mpg.de/tailornet/), CVPR 2020
* [深度时尚 3D：一个用于从单张图像进行 3D 服装重建的数据集与基准](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited/), ECCV 2020
* [SIZER：一个用于解析 3D 服装并学习尺寸敏感型 3D 服装的数据集与模型](https://virtualhumans.mpi-inf.mpg.de/sizer/), ECCV 2020

<!---
### People
* [Tamara Berg](http://www.tamaraberg.com/)
* [Kristen Graumen](http://www.cs.utexas.edu/users/grauman/)
* [Ranjitha Kumar](http://ranjithakumar.net/)
* [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/)
* [Kota Yamaguchi](https://sites.google.com/view/kyamagu/home)
-->

### 五花八门<a name='miscellaneous'></a>

- [Fashion-Gen：生成式时尚数据集及挑战](https://arxiv.org/abs/1806.08317v1)
- [品牌 -> 标志：时尚品牌的视觉分析](https://arxiv.org/pdf/1810.09941v1.pdf)

### 作者

- 作者: [Ayushi Dalmia](https://github.com/ayushidalmia)
- 翻译: [Turing Zhu](https://github.com/shamrock7)


### License

MIT

