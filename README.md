# EIE4512 Obtaining High-quality Panorama from Videos
This project is the final project of EIE4512 (Digital Image Processing) in the Chinese University of Hongkong, Shenzhen.

## Introduction
Panorama construction is an essential research topic in digital image processing. Researchers have done a lot on panorama construction methods, most of which focus on image stitching, but few consider video stitching. This project studies panorama construction from videos. In terms of frame selection, we proposed a binary search method based on the similarity between neighboring frames. We further achieved cylindrical projection to eliminate distortion of the panorama. Finally, The obtained panorama is deblurred by analyzing the camera’s motion.

<img src="report/images/pipeline.png" width="500" height="200">

Key points of the implementation are explained in details in the [report](https://github.com/thiefCat/EIE4512_pano_proj/blob/main/report/Group_8_report.pdf)

## How to run
The code for this project is in final_code folder. You can prepare the video by yourself or use our videos in the videos folder. Please note that the video should be recorded from left to right, and keep the recorded horizontal.

  ```bash
  git clone https://github.com/nikitakaraevv/pointnet
  cd pointnet/final_code
  python main.py 
  ```

## Sample Result
Matching two images by SIFT and RANSAC:

<img src="report/images/4_match_demo_134-154.png" width="300" height="300">


The image stitching result for a sequence of images by homography:

<img src="report/images/1.png" width="400" height="300">


The image stitching result for a sequence of images after cylindrical projection:

<img src="report/images/5.png" width="500" height="300">


Even longer panorama:

<img src="report/images/pano21_3.png" width="858" height="189">


Deblurred panorama:

<img src="report/images/pano21_6.png" width="623" height="190">


