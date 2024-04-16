#pragma once
#include <math.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

#define SENSOR_SELECT false // true是HK, false是DH

#define OPEN_DISTORT false // 开启畸变矫正(默认未开启)

#define FETCH_HOMO_MATRIX false // 获取homo矩阵(默认不获取，仅加载)

#define DISTENCE_SELECT 1 // 可设定为1,2,3 表示物体与相机距离

#define WIDTH 1080  // 后处理图像宽度(默认1080)

#define HEIGHT 720  // 后处理图像高度(默认720)

#define PI 3.14159

// 图像去畸变方法
void distort(Mat& vis, Mat& ir);

// 图像配准方法
void align(Mat source, Mat target, Mat& dst, int T_source_value, int T_target_value);

void myWarpPerspective(const cv::Mat& src, cv::Mat& dst, const cv::Mat& M, const cv::Size& size);

// Homo矩阵保存与加载
void saveMatToTxt(const cv::Mat& mat, const std::string& filename);

cv::Mat loadMatFromTxt(const std::string& filename, int rows, int cols, int type);



// 图像融合方法
Mat add_fusion(Mat& rgbImage, Mat& infraredImage);

Mat TIF_GRAY(const Mat& img_r, const Mat& img_v);

Mat TIF_RGB(const Mat& img_r, const Mat& img_v);

void TIF(Mat& img_v, Mat& img_r, Mat& dst);




class WaveTransform
{
public:
	Mat WDT(const Mat& _src, const string _wname, const int _level);//小波分解
	Mat IWDT(const Mat& _src, const string _wname, const int _level);//小波重构
	void wavelet_D(const string _wname, Mat& _lowFilter, Mat& _highFilter);//分解包
	void wavelet_R(const string _wname, Mat& _lowFilter, Mat& _highFilter);//重构包
	Mat waveletDecompose(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter);
	Mat waveletReconstruct(const Mat& _src, const Mat& _lowFilter, const Mat& _highFilter);
};
Mat WaveTransform::WDT(const Mat& _src, const string _wname, const int _level)
{
	Mat_<float> src = Mat_<float>(_src);
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	int row = src.rows;
	int col = src.cols;
	//高通低通滤波器
	Mat lowFilter;
	Mat highFilter;
	wavelet_D(_wname, lowFilter, highFilter);
	//小波变换
	int t = 1;

	while (t <= _level)
	{
		//先进行 行小波变换
		//#pragma omp parallel for
		for (int i = 0; i < row; i++)
		{
			//取出src中要处理的数据的一行
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

		//小波列变换
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
		//更新 
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

	//高低通滤波器
	Mat lowFilter;
	Mat highFilter;
	wavelet_R(_wname, lowFilter, highFilter);

	//小波变换
	int t = 1;
	int row = N / std::pow(2., _level - 1);
	int col = D / std::pow(2., _level - 1);

	while (row <= N && col <= D)
		//while(t<=_level)
	{
		//列逆变换
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

		//行逆变换
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

	//频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());

	filter2D(src, dst1, -1, lowFilter);
	filter2D(src, dst2, -1, highFilter);

	//下采样
	//数据拼接
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

	/// 插值;
	Mat Up1 = Mat::zeros(1, D, src.type());
	Mat Up2 = Mat::zeros(1, D, src.type());


	for (int i = 0, cnt = 0; i < D / 2; i++, cnt += 2)
	{
		Up1.at<float>(0, cnt) = src.at<float>(0, i);     ///< 前一半
		Up2.at<float>(0, cnt) = src.at<float>(0, i + D / 2); ///< 后一半
	}

	/// 前一半低通，后一半高通
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());
	filter2D(Up1, dst1, -1, lowFilter);
	filter2D(Up2, dst2, -1, highFilter);

	/// 结果相加
	dst1 = dst1 + dst2;
	return dst1;
}
