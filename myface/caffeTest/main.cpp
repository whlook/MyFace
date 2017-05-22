#include <stdio.h> 
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <thread>
#include <mutex>

#include "seeta_common.h"
#include "face_detection.h"
#include "face_identification.h"
#include "facedetect-dll.h"
#include <opencv2/opencv.hpp>
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#pragma comment(lib,"libfacedetect.lib")

#define NetTy float 

using namespace caffe;
using namespace seeta;
using std::cout;
using std::endl;
using std::string;
using std::vector;



template <typename Dtype>
caffe::Net<Dtype>* loadNet(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
	caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));

	net->CopyTrainedLayersFrom(pretrained_param_file);

	return net;
}



int main()
{
	cv::VideoCapture videoCap(0);
	cv::Mat img, face,img_gray;
	int * pResults = NULL;
	cv::Rect face_rect;
	short * p;
	int x, y, w, h;
	float * feat = new float[2048];
	float * wuhui = new float[2048];
	float similarity;
	int cnt = 0;

	FaceIdentification * face_recog = new FaceIdentification("seeta_fr_v1.0.bin");

	Caffe::set_mode(Caffe::CPU);
	caffe::Net<NetTy>* _net = loadNet<NetTy>((string)"face_deploy.prototxt", (string)"face.caffemodel", caffe::TEST); // 加载模型
	std::vector<caffe::Blob<NetTy>*> input_vec;  // 无意义

	while (1)
	{
		videoCap >> img;

		cv::cvtColor(img, img_gray,CV_BGR2GRAY);

		pResults = facedetect_frontal_tmp((unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, img_gray.step, 1.2f, 5, 24);
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			p = ((short*)(pResults + 1)) + 6 * i;  // 获取人脸信息

			x = p[0];
			y = p[1];
			w = p[2];
			h = p[3];

			if (!((x + w)>630 || x<5 || (y + h)>470 || y < 5)) // 防止人脸越界
			{
				face_rect.x = x; face_rect.y = y; face_rect.width = w; face_rect.height = h;

				cv::rectangle(img, face_rect, CV_RGB(0, 100, 200), 2, 8, 0);  // 画出人脸

				img(face_rect).copyTo(face);
				cv::resize(face, face, cv::Size(100, 100));

			}
		}
		std::vector<cv::Mat> dv = { face }; // AddMatVector(const vector<cv::Mat>& mat_vector,const vector<int>& labels)
		std::vector<int> label = { 0 };    // -------------------------------------------------------------------------

		caffe::MemoryDataLayer<NetTy> *m_layer = (caffe::MemoryDataLayer<NetTy> *)_net->layers()[0].get(); // 定义个内存数据层指针

		m_layer->AddMatVector(dv, label); // 这两行是使用MemoryData层必须的，这是把图片和标签，添加到 MemoryData层

		_net->Forward(input_vec);                    // 执行一次前向计算

		boost::shared_ptr<caffe::Blob<NetTy>> layerData = _net->blob_by_name("res5_6");  // 获得指定层的输出特征

		const float * feats = layerData->cpu_data(); // res5_6->cpu_data()返回的是多维数据（可以看成是个数组），

		for (int i = 0; i < 2048; ++i)
		{
			feat[i] = (*feats + *(feats + 1) + *(feats + 2) + *(feats + 3) + *(feats + 4) + *(feats + 5) + *(feats + 6)
				+ *(feats + 7) + *(feats + 8) + *(feats + 9)) / 10;
			feats+=10;
		}
		if (cnt == 0)
		{
			cnt = 1;
			for (int i = 0; i < 2048; ++i)
			{
				wuhui[i] = feat[i];
			}
		}
		for (int i = 0; i < 2048; ++i)
		{
			cout << wuhui[i] << "," << feat[i] << endl;
		}
		similarity = face_recog->CalcSimilarity(feat, wuhui);
		cout << similarity << endl;
		cv::imshow("face", img);
		cv::waitKey(10);
		
	}
	return 0;
}