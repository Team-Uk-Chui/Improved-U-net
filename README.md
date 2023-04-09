# Image Denoising Method based on Deep Learning using Improved U-net

한재욱, 최진원, 이창우  
가톨릭대학교 정보통신전자공학부

- [페이퍼 원문](https://paper.cricit.kr/user/listview/ieie2018/cart_rdoc.asp?URL=files/ieietspc_202108_001.pdf%3Fnum%3D408033%26db%3DRD_R&dn=408033&db=RD_R&usernum=0&seid=)

## 개요
본 모델은 이미지 노이즈 제거에 위한 모델이며, 기존 심층 신경망 중 전처리/후처리 과정을 추가하고 각 단계를 개선하여 이미지 복원에 널리 사용되는 U-net을 개선한 모델입니다. 

노이즈의 단계마다 따로 학습을 진행할 필요 없이 단일 학습만으로 광범위한 단계의 노이즈에 대해 기존의 모델들에 비해 높은 성능을 보입니다.

## 모델의 구조
<img src="https://i.imgur.com/Eaqv3wl.jpg"  width="400" height="700"/>

## 기존 모델과의 비교
![](https://i.imgur.com/lREpAo4.png)

![](https://i.imgur.com/LrbAxJM.jpg)

## 참조
<details>
  <summary>
    <b>참조</b>
  </summary>
    
[1] M. Mafi, S. Tabarestani, M. Cabrerizo, A. Barreto, and M. Adjouadi, “Denoising of ultrasound images affected by combined speckle and Gaussian noise,” IET Image Processing, vol. 12, np. 12, pp.2346–2351, 2018.  
[2] Y. Dong and S. Xu, “A new directional weighted median filter for removal of random-valued impulse noise,” IEEE Signal Processing Letters, vol. 14, no. 3, pp. 193–196, 2007.  
[3] A. Buades, B. Coll and J.-M. Morel, "A non-local algorithm for image denoising," in Proc. of Computer Vision and Pattern Recognition 2005 (CVPR 2005), pp. 60-65, June 2005.  
[4] K. Dabov, A. Foi and V. Katkovnik and K. Egiazarian, "Image denoising by sparse 3-D transform domain collaborative filtering," IEEE Trans. on Image Processing, vol. 16, no. 8, pp. 2080-2095, Aug. 2007.   
[5] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,” IEEE Transactions on Image Processing, 26(7): 3142–3155, 2017.  
[6] K. Zhang, W. Zuo, S. Gu, and L. Zhang, “Learning deep CNN denoiser prior for image restoration,” in CVPR 2017.  
[7] K. Zhang, W. Zuo, and L. Zhang, “FFDNet: Toward a fast and flexible solution for CNN-based image denoising,” IEEE Transactions on Image Processing, vol. 27, no. 9, pp. 4608–4622, 2018.  
[8] C. Tian, Y. Xu, Z. Li, W. Zuo, L. Fei and H. Liu, “Attention-guided CNN for image denoising,” Neural networks, vol. 124, pp. 117-129, Aprial 2020.  
[9] C. Tian, L. Fei, W. Zheng, Y. W. Zuo, C-W. Lin, “Deep learning on image denoising: An overview,” Neural networks, vol. 131, pp. 251-275, Nov. 2020.  
[10] O. Ronneberger, P. Fischer and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” MICCAI 2015: Medical Image Computing and Computer-Assisted Intervention 2015, pp. 234-241, 2015.  
[11] Y. J. Kim and C. W. Lee, “Deep Learning Method for Extending Image Intensity Using Hybrid Log-Gamma,” IEIE Transactions on Smart Processing and Computing, vol. 9, no. 4, pp. 312-316, August 2020.  
[12] H. Dong, A. Supratak, L. Mai, F. Liu, A. Oehmichen, S. Yu and Y. Guo, “TensorLayer: A versatile library for efficient deep learning development,” in Proc. ACM-MM 2017, pp. 1201–1204, 2017.  
[13] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in CVPRW 2017.  
[14] R. Franzen, “Kodak lossless true color image suite,” source: http://r0k.us/graphics/kodak, vol. 4, 1999.  
[15] D. Martin, C. Fowlkes, D. Tal, and J. Malik, “A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics,” in ICCV 2001.  
[16] D. Kingma and J. B. Adam, “Adam: A method for stochastic optimization,” International Conference on Learning Representations, 2015.  
[17] A. Horé and D. Ziou, “Image quality metrics: PSNR vs. SSIM,” 20th International Conference on Pattern Recognition, 2010.  

</details>
