#pragma once
#include <math.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

#define SENSOR_SELECT false // true��HK, false��DH

#define OPEN_DISTORT false // �����������(Ĭ��δ����)

#define FETCH_HOMO_MATRIX false // ��ȡhomo����(Ĭ�ϲ���ȡ��������)

#define DISTENCE_SELECT 1 // ���趨Ϊ1,2,3 ��ʾ�������������

#define WIDTH 1080  // ����ͼ����(Ĭ��1080)

#define HEIGHT 720  // ����ͼ��߶�(Ĭ��720)

#define PI 3.14159

// ͼ��ȥ���䷽��
void distort(Mat& vis, Mat& ir);

// ͼ����׼����
void align(Mat source, Mat target, Mat& dst, int T_source_value, int T_target_value);

void myWarpPerspective(const cv::Mat& src, cv::Mat& dst, const cv::Mat& M, const cv::Size& size);

// Homo���󱣴������
void saveMatToTxt(const cv::Mat& mat, const std::string& filename);

cv::Mat loadMatFromTxt(const std::string& filename, int rows, int cols, int type);



// ͼ���ںϷ���
Mat add_fusion(Mat& rgbImage, Mat& infraredImage);

Mat TIF_GRAY(const Mat& img_r, const Mat& img_v);

Mat TIF_RGB(const Mat& img_r, const Mat& img_v);

void TIF(Mat& img_v, Mat& img_r, Mat& dst);




class WaveTransform
{
public:
	Mat WDT(const Mat& _src, const string _wname, const int _level);//С���ֽ�
	Mat IWDT(const Mat& _src, const string _wname, const int _level);//С���ع�
	void wavelet_D(const string _wname, Mat& _lowFilter, Mat& _highFilter);//�ֽ��
	void wavelet_R(const string _wname, Mat& _lowFilter, Mat& _highFilter);//�ع���
	Mat waveletDecompose(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter);
	Mat waveletReconstruct(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter);
};
Mat WaveTransform::WDT(const Mat& _src, const string _wname, const int _level)
{
	Mat_<float> src = Mat_<float>(_src);
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	int row = src.rows;
	int col = src.cols;
	//��ͨ��ͨ�˲���
	Mat lowFilter;
	Mat highFilter;
	wavelet_D(_wname, lowFilter, highFilter);
	//С���任
	int t = 1;

	while (t <= _level)
	{
		//�Ƚ��� ��С���任
		//#pragma omp parallel for
		for (int i = 0; i < row; i++)
		{
			//ȡ��src��Ҫ��������ݵ�һ��
			Mat oneRow = Mat::zeros(1, col, src.type());

			for (int j = 0; j < col; j++)
			{
				oneRow.at<float>(0, j) = src.at<float>(i, j);
			}

			oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
			for (int j = 0; j < col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}

		//С���б任
		//#pragma omp parallel for
		for (int j = 0; j < col; j++)
		{
			Mat oneCol = Mat::zeros(row, 1, src.type());

			for (int i = 0; i < row; i++)
			{
				oneCol.at<float>(i, 0) = dst.at<float>(i, j);//dst,not src
			}
			oneCol = (waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();

			for (int i = 0; i < row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}
		}
		//���� 
		row /= 2;
		col /= 2;
		t++;
		src = dst;
	}
	return dst;
}

Mat WaveTransform::IWDT(const Mat& _src, const string _wname, const int _level)
{

	Mat src = Mat_<float>(_src);
	Mat dst;
	src.copyTo(dst);
	int N = src.rows;
	int D = src.cols;

	//�ߵ�ͨ�˲���
	Mat lowFilter;
	Mat highFilter;
	wavelet_R(_wname, lowFilter, highFilter);

	//С���任
	int t = 1;
	int row = N / std::pow(2., _level - 1);
	int col = D / std::pow(2., _level - 1);

	while (row <= N && col <= D)
		//while(t<=_level)
	{
		//����任
		for (int j = 0; j < col; j++)
		{
			Mat oneCol = Mat::zeros(row, 1, src.type());

			for (int i = 0; i < row; i++)
			{
				oneCol.at<float>(i, 0) = src.at<float>(i, j);
			}
			oneCol = (waveletReconstruct(oneCol.t(), lowFilter, highFilter)).t();

			for (int i = 0; i < row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}

		}

		//����任
		for (int i = 0; i < row; i++)
		{
			Mat oneRow = Mat::zeros(1, col, src.type());
			for (int j = 0; j < col; j++)
			{
				oneRow.at<float>(0, j) = dst.at<float>(i, j);
			}
			oneRow = waveletReconstruct(oneRow, lowFilter, highFilter);
			for (int j = 0; j < col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}

		row *= 2;
		col *= 2;
		t++;
		src = dst;
	}

	return dst;
}

void WaveTransform::wavelet_D(const string _wname, Mat& _lowFilter, Mat& _highFilter)
{
	if (_wname == "haar" || _wname == "db1")
	{
		int N = 2;
		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		_lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		_highFilter.at<float>(0, 0) = -1 / sqrtf(N);
		_highFilter.at<float>(0, 1) = 1 / sqrtf(N);
	}
	else if (_wname == "sym2")
	{
		int N = 4;
		float h[] = { -0.4830, 0.8365, -0.2241, -0.1294 };
		float l[] = { -0.1294, 0.2241,  0.8365, 0.4830 };

		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			_lowFilter.at<float>(0, i) = l[i];
			_highFilter.at<float>(0, i) = h[i];
		}
	}
}

void WaveTransform::wavelet_R(const string _wname, Mat& _lowFilter, Mat& _highFilter)
{
	if (_wname == "haar" || _wname == "db1")
	{
		int N = 2;
		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);


		_lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		_highFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_highFilter.at<float>(0, 1) = -1 / sqrtf(N);
	}
	else if (_wname == "sym2")
	{
		int N = 4;
		float h[] = { -0.1294,-0.2241,0.8365,-0.4830 };
		float l[] = { 0.4830, 0.8365, 0.2241, -0.1294 };

		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			_lowFilter.at<float>(0, i) = l[i];
			_highFilter.at<float>(0, i) = h[i];
		}
	}
}

Mat WaveTransform::waveletDecompose(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	//Mat& src = Mat_<float>(_src);
	Mat_<float>src = _src;

	int D = src.cols;

	/*Mat& lowFilter = Mat_<float>(_lowFilter);
	Mat& highFilter = Mat_<float>(_highFilter);*/

	Mat_<float>lowFilter = _lowFilter;
	Mat_<float>highFilter = _highFilter;

	//Ƶ���˲���ʱ������ifft( fft(x) * fft(filter)) = cov(x,filter) 
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());

	filter2D(src, dst1, -1, lowFilter);
	filter2D(src, dst2, -1, highFilter);

	//�²���
	//����ƴ��
	for (int i = 0, j = 1; i < D / 2; i++, j += 2)
	{
		src.at<float>(0, i) = dst1.at<float>(0, j);//lowFilter
		src.at<float>(0, i + D / 2) = dst2.at<float>(0, j);//highFilter
	}
	return src;
}

Mat WaveTransform::waveletReconstruct(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	//Mat& src = Mat_<float>(_src);
	Mat_<float>src = _src;

	int D = src.cols;

	/*Mat& lowFilter = Mat_<float>(_lowFilter);
	Mat& highFilter = Mat_<float>(_highFilter);*/
	Mat_<float>lowFilter = _lowFilter;
	Mat_<float>highFilter = _highFilter;

	/// ��ֵ;
	Mat Up1 = Mat::zeros(1, D, src.type());
	Mat Up2 = Mat::zeros(1, D, src.type());


	for (int i = 0, cnt = 0; i < D / 2; i++, cnt += 2)
	{
		Up1.at<float>(0, cnt) = src.at<float>(0, i);     ///< ǰһ��
		Up2.at<float>(0, cnt) = src.at<float>(0, i + D / 2); ///< ��һ��
	}

	/// ǰһ���ͨ����һ���ͨ
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());
	filter2D(Up1, dst1, -1, lowFilter);
	filter2D(Up2, dst2, -1, highFilter);

	/// ������
	dst1 = dst1 + dst2;
	return dst1;
}
