#include "main.h"

// 畸变矫正
void distort(Mat& vis, Mat& ir)
{
	if (SENSOR_SELECT)
	{
		// TODO: 海康相机暂未标定
	}
	else
	{
		/*********************************大华RGB相机畸变参数**********************************/
		// fx, fy, cx, cy为实际数值
		cv::Mat cameraMatrix_vis = (cv::Mat_<double>(3, 3) << 1563.7, 0, 957.0,
			0, 1173.0, 548.9,
			0, 0, 1);
		// k1, k2, p1, p2, k3为实际数值
		cv::Mat distCoeffs_vis = (cv::Mat_<double>(1, 5) << -0.4312, 0.1983, -0.0023, 0.0022, 0.0);
		//去畸变
		cv::Mat vis_distort;
		cv::undistort(vis, vis_distort, cameraMatrix_vis, distCoeffs_vis);
		vis = vis_distort.clone();
		/*********************************大华IR相机畸变参数**********************************/
		cv::Mat cameraMatrix_ir = (cv::Mat_<double>(3, 3) << 293.2257, 0, 149.3663,
			0, 293.7956, 115.4141,
			0, 0, 1);
		cv::Mat distCoeffs_ir = (cv::Mat_<double>(1, 5) << -0.4134, 0.1828, -0.0035, 0.0007, 0.0);
		//去畸变
		Mat ir_distort;
		cv::undistort(ir, ir_distort, cameraMatrix_ir, distCoeffs_ir);
		ir = ir_distort.clone();
		/**************************************************************************************/
	}
}

// Homo矩阵保存
void saveMatToTxt(const cv::Mat& mat, const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "无法打开文件：" << filename << std::endl;
		return;
	}

	// 假设mat是一个3x3的浮点矩阵
	if (mat.rows != 3 || mat.cols != 3 || mat.type() != CV_32FC1)
	{
		std::cerr << "矩阵大小或类型不正确。" << std::endl;
		return;
	}

	// 遍历矩阵并写入文件
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			file << mat.at<float>(i, j) << " ";
		}
		file << std::endl; // 每行结束后换行
	}

	file.close();
}

// Homo矩阵加载
cv::Mat loadMatFromTxt(const std::string& filename, int rows, int cols, int type)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "无法打开文件：" << filename << std::endl;
		return cv::Mat();
	}

	cv::Mat mat(rows, cols, type);
	std::string line;
	int row = 0;

	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		float value;
		int col = 0;

		while (iss >> value)
		{
			mat.at<float>(row, col++) = value;
		}

		row++;
		if (row >= rows)
		{
			break; // 如果已经读取了足够的行数，就退出循环
		}
	}

	file.close();
	return mat;
}

// 透视变换
void myWarpPerspective(const cv::Mat& src, cv::Mat& dst, const cv::Mat& M, const cv::Size& size)
{
	if (src.channels() == 1)
	{
		dst = cv::Mat::zeros(size, CV_8UC1); // 创建一个与目标大小相同的空白灰度图
		cv::Mat Minv = M.inv(cv::DECOMP_SVD);  // 使用SVD分解求逆以提高精度

		// 确保Minv是double类型
		if (Minv.type() != CV_64F) {
			Minv.convertTo(Minv, CV_64F);
		}

		for (int y = 0; y < size.height; ++y) 
		{
			for (int x = 0; x < size.width; ++x) 
			{
				// 使用double类型创建列向量
				cv::Mat pt = (cv::Mat_<double>(3, 1) << (double)x, (double)y, 1.0);

				// 矩阵乘法
				cv::Mat pt_prime = Minv * pt;
				double u = pt_prime.at<double>(0) / pt_prime.at<double>(2);
				double v = pt_prime.at<double>(1) / pt_prime.at<double>(2);

				// 双线性插值
				if (u >= 0 && u < src.cols && v >= 0 && v < src.rows) {
					int u1 = (int)u;
					int v1 = (int)v;
					int u2 = std::min(u1 + 1, src.cols - 1);
					int v2 = std::min(v1 + 1, src.rows - 1);
					float a = u - u1;
					float b = v - v1;

					uchar p1 = src.at<uchar>(v1, u1);
					uchar p2 = src.at<uchar>(v1, u2);
					uchar p3 = src.at<uchar>(v2, u1);
					uchar p4 = src.at<uchar>(v2, u2);

					uchar result = cv::saturate_cast<uchar>((1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4);
					dst.at<uchar>(y, x) = result;
				}
			}
		}
	}
	else
	{
		dst = cv::Mat::zeros(size, src.type());
		cv::Mat Minv = M.inv(cv::DECOMP_SVD);  // 使用SVD分解求逆以提高精度

		// 确保Minv是double类型
		if (Minv.type() != CV_64F)
		{
			Minv.convertTo(Minv, CV_64F);
		}

		for (int y = 0; y < size.height; ++y)
		{
			for (int x = 0; x < size.width; ++x)
			{
				// 使用double类型创建列向量
				cv::Mat pt = (cv::Mat_<double>(3, 1) << (double)x, (double)y, 1.0);

				// 矩阵乘法
				cv::Mat pt_prime = Minv * pt;
				double u = pt_prime.at<double>(0) / pt_prime.at<double>(2);
				double v = pt_prime.at<double>(1) / pt_prime.at<double>(2);

				// 双线性插值
				if (u >= 0 && u < src.cols && v >= 0 && v < src.rows)
				{
					int u1 = (int)u;
					int v1 = (int)v;
					int u2 = std::min(u1 + 1, src.cols - 1);
					int v2 = std::min(v1 + 1, src.rows - 1);
					float a = u - u1;
					float b = v - v1;

					cv::Vec3b p1 = src.at<cv::Vec3b>(v1, u1);
					cv::Vec3b p2 = src.at<cv::Vec3b>(v1, u2);
					cv::Vec3b p3 = src.at<cv::Vec3b>(v2, u1);
					cv::Vec3b p4 = src.at<cv::Vec3b>(v2, u2);

					cv::Vec3b result = (1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4;
					dst.at<cv::Vec3b>(y, x) = result;
				}
			}
		}
	}
}

// 配准
void align(Mat source, Mat target, Mat& dst)
{
	if (FETCH_HOMO_MATRIX)
	{
		/**********************************************************************************/
		vector<Point2f> points1, points2;
		if (SENSOR_SELECT)
		{
			/**********************************海康相机****************************************/
			if (DISTENCE_SELECT == 1) // 1m距离
			{
				points1 = { Point2f(150,45),
							Point2f(156,56),Point2f(156,65),Point2f(161,144),Point2f(162,153),Point2f(211,54),
							Point2f(211,62),Point2f(213,70),Point2f(213,78),Point2f(215,86),Point2f(215,95),Point2f(216,103),
							Point2f(217,111),Point2f(217,119),Point2f(218,128),Point2f(219,136),Point2f(219,145) };
				points2 = { Point2f(152,40),Point2f(162,55),
								Point2f(162,64),Point2f(170,154),Point2f(171,164),Point2f(244,52),Point2f(245,62),
								Point2f(247,71),Point2f(248,80),Point2f(249,90),Point2f(249,99),Point2f(252,109),
								Point2f(252,118),Point2f(254,128),Point2f(254,138),Point2f(255,147),Point2f(255,157) };
			}
			else if (DISTENCE_SELECT == 2) // 2m距离
			{
				points1 = {
				Point2f(150,88),Point2f(153,93),Point2f(153,98),Point2f(155,135),Point2f(155,141),
				Point2f(181,90),Point2f(181,95),Point2f(184,132),Point2f(185,138),Point2f(183,84), };
				points2 = {
				Point2f(144,90),Point2f(148,97),Point2f(148,102),Point2f(151,144),Point2f(152,150),
				Point2f(192,94),Point2f(192,99),Point2f(195,142),Point2f(195,147),Point2f(195,87), };
			}
			else // 3m 距离
			{
				points1 = { Point2f(144,80),Point2f(128,99),Point2f(147,125),Point2f(163,106)};
				points2 = { Point2f(132,79),Point2f(107,103),Point2f(138,135),Point2f(164,111)};
			}
		}
		else
		{
			/**********************************大华相机****************************************/
			if (DISTENCE_SELECT == 1) // 1m距离
			{
				points1 = {
				Point2f(90,58),Point2f(92,65),Point2f(91,71),Point2f(87,120),Point2f(87,126),
				Point2f(141,67),Point2f(141,73),Point2f(138,124),Point2f(138,130),Point2f(146,60)};
				points2 = {
				Point2f(102,55),Point2f(106,66),Point2f(106,74),Point2f(101,142),Point2f(100,151),
				Point2f(174,68),Point2f(174,77),Point2f(170,147),Point2f(170,155),Point2f(181,57)};
			}
			else if (DISTENCE_SELECT == 2) // 2m距离
			{
				points1 = { Point2f(138,76),Point2f(118,95),Point2f(141,120),Point2f(162,102)};
				points2 = { Point2f(165,81),Point2f(137,107),Point2f(170,142),Point2f(199,116)};
			}
			else // 3m距离
			{
				points1 = { Point2f(139,86),Point2f(124,96),Point2f(136,115),Point2f(152,105)};
				points2 = { Point2f(164,95),Point2f(143,109),Point2f(161,136),Point2f(183,121)};
			}
		}


		// 计算Homo矩阵
		Mat Homo = findHomography(points1, points2, RANSAC); // 将可见光变换至红外
		//Mat Homo = findHomography(points2, points1, RANSAC); // 将红外变换至可见光

		Homo.convertTo(Homo, CV_32FC1);


		// 利用Homo矩阵做透视变换
		//cv::warpPerspective(source, dst, Homo, target.size()); // 将可见光变换至红外
		//cv::warpPerspective(target, dst, Homo, source.size()); // 将红外变换至可见光
		myWarpPerspective(source, dst, Homo, target.size());     // 重写版

		// 保存到文本文件
		if (SENSOR_SELECT)
		{
			if (DISTENCE_SELECT == 1)
			{
				saveMatToTxt(Homo, "Homo_hk_1m.txt");
			}
			else if (DISTENCE_SELECT == 2)
			{
				saveMatToTxt(Homo, "Homo_hk_2m.txt");
			}
			else
			{
				saveMatToTxt(Homo, "Homo_hk_3m.txt");
			}
		}
		else
		{
			if (DISTENCE_SELECT == 1)
			{
				saveMatToTxt(Homo, "Homo_dh_1m.txt");
			}
			else if (DISTENCE_SELECT == 2)
			{
				saveMatToTxt(Homo, "Homo_dh_2m.txt");
			}
			else
			{
				saveMatToTxt(Homo, "Homo_dh_3m.txt");
			}
		}
	}
	else
	{
		// 加载Homo矩阵
		Mat Homo;
		if (SENSOR_SELECT)
		{
			if (DISTENCE_SELECT == 1)
			{
				Homo = loadMatFromTxt("Homo_hk_1m.txt", 3, 3, CV_32FC1);
			}
			else if (DISTENCE_SELECT == 2)
			{
				Homo = loadMatFromTxt("Homo_hk_2m.txt", 3, 3, CV_32FC1);
			}
			else
			{
				Homo = loadMatFromTxt("Homo_hk_3m.txt", 3, 3, CV_32FC1);
			}
		}
		else
		{
			if (DISTENCE_SELECT == 1)
			{
				Homo = loadMatFromTxt("Homo_dh_1m.txt", 3, 3, CV_32FC1);
			}
			else if (DISTENCE_SELECT == 2)
			{
				Homo = loadMatFromTxt("Homo_dh_2m.txt", 3, 3, CV_32FC1);
			}
			else
			{
				Homo = loadMatFromTxt("Homo_dh_3m.txt", 3, 3, CV_32FC1);
			}
		}

		// 透视变换
		myWarpPerspective(source, dst, Homo, target.size());
	}
}

// 融合
/*************************wave_fusion*****************************/
void RGB2HSI(Mat src, Mat& dst)
{
	Mat HSI(src.rows, src.cols, CV_32FC3);
	float r, g, b, H, S, I, num, den, theta, sum, min_RGB;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = src.at<Vec3b>(i, j)[0];
			g = src.at<Vec3b>(i, j)[1];
			r = src.at<Vec3b>(i, j)[2];

			num = 0.5 * ((r - g) + (r - b));
			den = sqrt((r - g) * (r - g) + (r - b) * (g - b));

			if (den == 0)
			{
				H = 0; // 分母不能为0
			}
			else
			{
				theta = acos(num / den);
				if (b <= g)
				{
					H = theta;
				}
				else
				{
					H = (2 * PI - theta);
				}
			}

			min_RGB = min(min(b, g), r); // min(R,G,B)
			sum = b + g + r;
			if (sum == 0)
			{
				S = 0;
			}
			else
			{
				S = 1 - 3 * min_RGB / sum;
			}

			I = sum / 3.0;

			HSI.at<Vec3f>(i, j)[0] = H;
			HSI.at<Vec3f>(i, j)[1] = S;
			HSI.at<Vec3f>(i, j)[2] = I;
		}
	}
	dst = HSI;
	return;
}
void HSI2RGB(Mat src, Mat& dst)
{
	Mat RGB(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			float DH = src.at<Vec3f>(i, j)[0];
			float DS = src.at<Vec3f>(i, j)[1];
			float DI = src.at<Vec3f>(i, j)[2];
			//分扇区显示
			float R, G, B;
			if (DH < (2 * PI / 3) && DH >= 0)
			{
				B = DI * (1 - DS);
				R = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
				G = (3 * DI - (R + B));
			}
			else if (DH < (4 * PI / 3) && DH >= (2 * PI / 3))
			{
				DH = DH - (2 * PI / 3);
				R = DI * (1 - DS);
				G = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
				B = (3 * DI - (G + R));
			}
			else
			{
				DH = DH - (4 * PI / 3);
				G = DI * (1 - DS);
				B = DI * (1 + (DS * cos(DH)) / cos(PI / 3 - DH));
				R = (3 * DI - (G + B));
			}
			RGB.at<Vec3f>(i, j)[0] = B;
			RGB.at<Vec3f>(i, j)[1] = G;
			RGB.at<Vec3f>(i, j)[2] = R;
		}
	}
	dst = RGB;
	return;
}
void harr_fusion(Mat src1, Mat src2, Mat& dst)
{
	assert(src1.rows == src2.rows && src1.cols == src2.cols);
	int row = src1.rows;
	int col = src1.cols;
	Mat src1_gray, src2_gray;
	normalize(src1, src1_gray, 0, 255, NORM_MINMAX);
	cvtColor(src2, src2_gray, COLOR_RGB2GRAY);
	normalize(src2_gray, src2_gray, 0, 255, NORM_MINMAX);
	src1_gray.convertTo(src1_gray, CV_32F);
	src2_gray.convertTo(src2_gray, CV_32F);
	WaveTransform m_waveTransform;
	const int level = 1;
	Mat src1_dec = m_waveTransform.WDT(src1_gray, "haar", level);
	Mat src2_dec = m_waveTransform.WDT(src2_gray, "haar", level);
	Mat dec = Mat(row, col, CV_32FC1);
	//融合规则：高频部分采用加权平均的方法，低频部分采用模值取大的方法
	int halfRow = row / (2 * level);
	int halfCol = col / (2 * level);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (i < halfRow && j < halfCol)
			{
				dec.at<float>(i, j) = (src1_dec.at<float>(i, j) + src2_dec.at<float>(i, j)) / 2;
			}
			else
			{
				float p = abs(src1_dec.at<float>(i, j));
				float q = abs(src2_dec.at<float>(i, j));
				if (p > q)
				{
					dec.at<float>(i, j) = src1_dec.at<float>(i, j);
				}
				else
				{
					dec.at<float>(i, j) = src2_dec.at<float>(i, j);
				}

			}
		}
	}
	dst = m_waveTransform.IWDT(dec, "haar", level);
}
void fusion(Mat Visible, Mat Infrared, Mat& dst)
{
	Mat HSI(Visible.size(), CV_32FC3);
	Mat Visible_I(Visible.size(), CV_32FC1);
	RGB2HSI(Visible, HSI);
	for (int i = 0; i < Visible.rows; i++)
	{
		for (int j = 0; j < Visible.cols; j++)
		{
			Visible_I.at<float>(i, j) = HSI.at<Vec3f>(i, j)[2];
		}
	}
	Mat fusion_I;
	harr_fusion(Visible_I, Infrared, fusion_I);
	Mat fusion_dst = Mat::zeros(Visible.size(), CV_32FC3);
	for (int i = 0; i < Visible.rows; i++)
	{
		for (int j = 0; j < Visible.cols; j++)
		{
			fusion_dst.at<Vec3f>(i, j)[2] = fusion_I.at<float>(i, j);
			fusion_dst.at<Vec3f>(i, j)[0] = HSI.at<Vec3f>(i, j)[0];
			fusion_dst.at<Vec3f>(i, j)[1] = HSI.at<Vec3f>(i, j)[1];
		}
	}
	HSI2RGB(fusion_dst, dst);
	dst.convertTo(dst, CV_8UC3);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
}

/*************************add_fusion*****************************/
Mat add_fusion(Mat& rgbImage, Mat& infraredImage)
{
	// 将RGB图像和红外图像转换为灰度图像
	Mat grayRgbImage, grayInfraredImage;
	/*cvtColor(rgbImage, grayRgbImage, COLOR_BGR2GRAY);
	cvtColor(infraredImage, grayInfraredImage, COLOR_BGR2GRAY);*/

	grayRgbImage = rgbImage;
	grayInfraredImage = infraredImage;

	// 将图像数据类型转换为浮点类型
	Mat grayRgbImageFloat, grayInfraredImageFloat;
	grayRgbImage.convertTo(grayRgbImageFloat, CV_64F);
	grayInfraredImage.convertTo(grayInfraredImageFloat, CV_64F);


	// 设置权重并计算加权平均
	double weightRgb = 0.5, weightInfrared = 0.5;
	Mat fusedImage = weightRgb * grayRgbImageFloat + weightInfrared * grayInfraredImageFloat;

	fusedImage.convertTo(fusedImage, CV_8U);

	return fusedImage;
}

/*************************TIF_fusion*****************************/
Mat TIF_GRAY(const Mat& img_r, const Mat& img_v)
{
	Mat img_r_blur, img_v_blur;
	/*blur(img_r, img_r_blur, Size(35, 35));
	blur(img_v, img_v_blur, Size(35, 35));*/
	blur(img_r, img_r_blur, Size(3, 3));
	blur(img_v, img_v_blur, Size(3, 3));

	Mat img_r_median, img_v_median;
	medianBlur(img_r, img_r_median, 3);
	medianBlur(img_v, img_v_median, 3);

	Mat img_r_detail = img_r - img_r_blur;
	Mat img_v_detail = img_v - img_v_blur;

	Mat img_r_the, img_v_the;
	pow(abs(img_r_median - img_r_blur), 2, img_r_the);
	pow(abs(img_v_median - img_v_blur), 2, img_v_the);

	Mat img_r_weight = img_r_the / (img_r_the + img_v_the + 0.000001);
	Mat img_v_weight = 1 - img_r_weight;

	Mat img_base_fused = (img_r_blur + img_v_blur) / 2;
	Mat img_detail_fused = img_r_weight.mul(img_r_detail) + img_v_weight.mul(img_v_detail);

	Mat img_fused_tmp = img_base_fused + img_detail_fused;
	//img_fused_tmp.convertTo(img_fused_tmp, CV_32S); // Ensure the type is int for thresholding

	img_fused_tmp.convertTo(img_fused_tmp, CV_32F); // 转换为浮点数以便处理

	// 保证值在0到255之间，然后转换为CV_8U
	normalize(img_fused_tmp, img_fused_tmp, 0, 255, NORM_MINMAX);


	//// Thresholding to keep pixel values in [0, 255]
	//threshold(img_fused_tmp, img_fused_tmp, 255, 255, THRESH_TRUNC);
	//threshold(img_fused_tmp, img_fused_tmp, 0, 0, THRESH_TOZERO);

	Mat img_fused;
	img_fused_tmp.convertTo(img_fused, CV_8U);

	return img_fused;
}

Mat TIF_RGB(const Mat& img_r, const Mat& img_v)
{
	vector<Mat> channels_r, channels_v, fused_channels;

	split(img_r, channels_r);
	split(img_v, channels_v);

	for (int i = 0; i < 3; ++i)
	{
		fused_channels.push_back(TIF_GRAY(channels_r[i], channels_v[i]));
	}

	Mat fused_img;
	merge(fused_channels, fused_img);

	return fused_img;
}

void TIF(Mat& img_v, Mat& img_r, Mat& dst)
{
	/*Mat img_r = imread(rpath);
	Mat img_v = imread(vpath);*/

	if (img_r.empty())
	{
		cout << "img_r is null" << endl;
		return;
	}
	if (img_v.empty())
	{
		cout << "img_v is null" << endl;
		return;
	}
	if (img_r.rows != img_v.rows || img_r.cols != img_v.cols)
	{
		cout << "size is not equal" << endl;
		return;
	}

	//Mat fused_img;
	if (img_r.channels() < 3 || img_r.channels() == 1)
	{
		if (img_v.channels() < 3 || img_v.channels() == 1)
		{
			dst = TIF_GRAY(img_r, img_v);
		}
		else
		{
			Mat img_v_gray;
			cvtColor(img_v, img_v_gray, COLOR_BGR2GRAY);
			dst = TIF_GRAY(img_r, img_v_gray);
		}
	}
	else
	{
		if (img_v.channels() < 3 || img_v.channels() == 1)
		{
			Mat img_r_gray;
			cvtColor(img_r, img_r_gray, COLOR_BGR2GRAY);
			dst = TIF_GRAY(img_r_gray, img_v);
		}
		else
		{
			dst = TIF_RGB(img_r, img_v);
		}
	}
}

/************************IFEVIP_fusion***************************/
cv::Mat matrixMultiply(const cv::Mat& a, const cv::Mat& b)
{
	cv::Mat result;
	cv::gemm(a, b, 1.0, cv::Mat(), 0.0, result);
	return result;
}

cv::Mat QuadReconstructRefined(const cv::Mat& S, const cv::Mat& img, int minDim) 
{
	cv::Mat imgDouble;
	img.convertTo(imgDouble, CV_64F);

	// Matrix M for Bézier interpolation
	cv::Mat M = (cv::Mat_<double>(4, 4) << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 3, 0, 0,
		1, 0, 0, 0);

	// Convert S to positive indices and compute maximum dimension
	cv::Mat newS = S + 1;
	double maxDim;
	cv::minMaxIdx(newS, nullptr, &maxDim);
	int dim = static_cast<int>(maxDim);

	// Pad image for computation
	cv::Mat newImg;
	cv::copyMakeBorder(imgDouble, newImg, 1, 1, 1, 1, cv::BORDER_REPLICATE);

	cv::Mat tempReconstImg = cv::Mat::zeros(newImg.size(), CV_64F);

	while (dim >= minDim + 1) 
	{
		for (int y = 1; y < newS.rows - 1; y++)
		{
			for (int x = 1; x < newS.cols - 1; x++)
			{
				if (newS.at<int>(y, x) == dim)
				{
					int subDim = (dim - 1) / 4;
					std::vector<cv::Point> pts = 
					{
						{x, y},
						{x + subDim * 2, y},
						{x + subDim * 3, y},
						{x + subDim * 4, y}
					};

					cv::Mat block = newImg(cv::Rect(x, y, dim, dim));
					cv::Mat blockVal = block.clone();  // Extract block

					// Generate u and v
					cv::Mat u = cv::Mat::zeros(dim, 1, CV_64F);
					for (int i = 0; i < dim; i++) 
					{
						u.at<double>(i, 0) = double(i) / (dim - 1);
					}
					cv::Mat U = cv::Mat::zeros(dim, 4, CV_64F);
					for (int i = 0; i < dim; i++) 
					{
						double ui = u.at<double>(i, 0);
						U.at<double>(i, 0) = ui * ui * ui;
						U.at<double>(i, 1) = ui * ui;
						U.at<double>(i, 2) = ui;
						U.at<double>(i, 3) = 1;
					}

					// Compute LM and RM
					cv::Mat LM = matrixMultiply(U, M);
					cv::Mat RM = matrixMultiply(M, U.t());

					// Reconstruct block
					cv::Mat reblkVal = matrixMultiply(matrixMultiply(LM, blockVal), RM);
					reblkVal.copyTo(tempReconstImg(cv::Rect(x, y, dim, dim)));
				}
			}
		}
		dim = (dim - 1) / 2 + 1;
	}

	// Crop the final reconstructed image to remove padding
	cv::Mat bgImg = tempReconstImg(cv::Rect(1, 1, img.cols, img.rows));
	return bgImg;
}

// 计算图像熵(简单实现)
double calculateEntropy(const cv::Mat& img) 
{
	std::vector<int> histogram(256, 0);
	for (int y = 0; y < img.rows; y++) 
	{
		for (int x = 0; x < img.cols; x++) 
		{
			histogram[img.at<uchar>(y, x)]++;
		}
	}
	double entropy = 0.0, total = img.rows * img.cols;
	for (int count : histogram) 
	{
		if (count > 0) 
		{
			double probability = count / total;
			entropy -= probability * log2(probability);
		}
	}
	return entropy;
}

cv::Mat BGR_Fuse(const cv::Mat& imgVis, const cv::Mat& imgIR, int QuadNormDim, int QuadMinDim, int GaussScale, double MaxRatio, double StdRatio) 
{
	// Resize IR image to a normalized dimension
	cv::Mat I;
	cv::resize(imgIR, I, cv::Size(QuadNormDim, QuadNormDim));

	// Simulate quadtree decomposition (using previous C++ implementation)
	cv::Mat S = cv::Mat::ones(I.size(), CV_32S);  // Placeholder: actual quadtree required
	cv::Mat minImg;
	cv::erode(I, minImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(QuadMinDim, QuadMinDim)));
	cv::Mat bgImg = QuadReconstructRefined(S, minImg, QuadMinDim);
	cv::resize(bgImg, bgImg, imgIR.size());

	// Gaussian filtering
	cv::GaussianBlur(bgImg, bgImg, cv::Size(GaussScale, GaussScale), GaussScale / 2.0, GaussScale / 2.0);

	// Extract infrared bright features
	cout << "imgIR.chanels:" << imgIR.channels() << endl;
	cout << "bgImg.chanels:" << bgImg.channels() << endl;
	bgImg.convertTo(bgImg, CV_8U);

	cv::Mat img_Inf = imgIR - bgImg;

	// Entropy based feature fusion
	double entropyIR = calculateEntropy(imgIR);
	double entropyVis = calculateEntropy(imgVis);
	cv::Mat addFeature = cv::Mat::zeros(imgVis.size(), CV_64F);
	addFeature = img_Inf + (imgIR - imgVis) * (entropyIR / entropyVis);

	// Gray level driven feature fusion
	cv::Mat addedVals = cv::max(addFeature, 0.0) + imgVis;
	std::vector<double> vals;
	addedVals.reshape(1, addedVals.total()).copyTo(vals);
	std::sort(vals.begin(), vals.end(), std::greater<double>());
	double maxMean = std::accumulate(vals.begin(), vals.begin() + int(MaxRatio * vals.size()), 0.0) / int(MaxRatio * vals.size());
	double ratio = std::min(255.0 / maxMean, StdRatio);

	cv::Mat result;
	cv::add(imgVis, addFeature * ratio, result, cv::Mat(), CV_8U);

	return result;
}


/**************************GFF_fusion****************************/
Mat guidedFilter(const Mat& img_i, const Mat& img_p, int r, double eps) 
{
	int wsize = 2 * r + 1;
	Mat meanI, meanP, corrI, corrIP, varI, covIP, a, b, meanA, meanB, q;

	boxFilter(img_i, meanI, CV_32F, Size(wsize, wsize));
	boxFilter(img_p, meanP, CV_32F, Size(wsize, wsize));
	boxFilter(img_i.mul(img_i), corrI, CV_32F, Size(wsize, wsize));
	boxFilter(img_i.mul(img_p), corrIP, CV_32F, Size(wsize, wsize));

	varI = corrI - meanI.mul(meanI);
	covIP = corrIP - meanI.mul(meanP);

	a = covIP / (varI + eps);
	b = meanP - a.mul(meanI);

	boxFilter(a, meanA, CV_32F, Size(wsize, wsize));
	boxFilter(b, meanB, CV_32F, Size(wsize, wsize));

	q = meanA.mul(img_i) + meanB;
	return q;
}

Mat GFF_GRAY(const Mat& img_v, const Mat& img_r)
{
	Mat img_r_float, img_v_float;
	img_r.convertTo(img_r_float, CV_32F, 1.0 / 255.0);
	img_v.convertTo(img_v_float, CV_32F, 1.0 / 255.0);

	Mat img_r_blur, img_v_blur;
	GaussianBlur(img_r_float, img_r_blur, Size(31, 31), 0);
	GaussianBlur(img_v_float, img_v_blur, Size(31, 31), 0);

	Mat img_r_detail = img_r_float - img_r_blur;
	Mat img_v_detail = img_v_float - img_v_blur;

	Mat img_r_lap, img_v_lap;
	Laplacian(img_r_float, img_r_lap, CV_32F, 3);
	Laplacian(img_v_float, img_v_lap, CV_32F, 3);

	int win_size = 2 * 5 + 1; // Using R_G from Python code
	Mat s1, s2;
	GaussianBlur(abs(img_r_lap), s1, Size(win_size, win_size), 5);
	GaussianBlur(abs(img_v_lap), s2, Size(win_size, win_size), 5);

	Mat p1 = Mat::zeros(img_r.size(), CV_32F);
	Mat p2 = Mat::zeros(img_v.size(), CV_32F);

	p1.setTo(1, s1 > s2);
	p2.setTo(1, s1 <= s2);

	Mat w1_b = guidedFilter(p1, img_r_float, 45, 0.3);
	Mat w2_b = guidedFilter(p2, img_v_float, 45, 0.3);
	Mat w1_d = guidedFilter(p1, img_r_float, 7, 0.000001);
	Mat w2_d = guidedFilter(p2, img_v_float, 7, 0.000001);

	Mat w1_b_w = w1_b / (w1_b + w2_b);
	Mat w2_b_w = w2_b / (w1_b + w2_b);
	Mat w1_d_w = w1_d / (w1_d + w2_d);
	Mat w2_d_w = w2_d / (w1_d + w2_d);

	Mat fused_b = w1_b_w.mul(img_r_blur) + w2_b_w.mul(img_v_blur);
	Mat fused_d = w1_d_w.mul(img_r_detail) + w2_d_w.mul(img_v_detail);

	Mat img_fused = fused_b + fused_d;
	normalize(img_fused, img_fused, 0, 255, NORM_MINMAX);

	Mat img_fused_abs;
	convertScaleAbs(img_fused, img_fused_abs); 
	return img_fused_abs;
}


int main(int argc, char* argv[])
{
	// 读取可见光与红外图片对
	Mat Visible = imread("D:/0412-DH/RGB_0.8-1m/10_RGB.jpg", IMREAD_GRAYSCALE); //待配准图像(可见光)
	Mat Infrared = imread("D:/0412-DH/IR_0.8-1m/10_IR.jpg",IMREAD_GRAYSCALE);  //目标图像(红外)

	// 判断图像是否为空
	if (Visible.empty() || Infrared.empty())
	{
		cout << "could not load the image..." << endl;
		return -1;
	}
	
	// 畸变矫正
	if (OPEN_DISTORT)
	{
		distort(Visible, Infrared);
	}
	
	// 如果大小不一致，将可见光大小resize至红外大小(要指定INTER_AREA方法，否则锯齿严重!)
	if (Visible.size() != Infrared.size())
	{
		cv::resize(Visible, Visible, Infrared.size(), 0, 0, cv::INTER_AREA);
	}
	

	// 转灰度图像
	/*cvtColor(Visible, Visible, COLOR_BGR2GRAY);
	cvtColor(Infrared, Infrared, COLOR_BGR2GRAY);*/

	// 图像配准(将RGB变换至IR视角)
	Mat vis_align;
	align(Visible, Infrared, vis_align);

	if (vis_align.data)
	{
		namedWindow("配准结果", WINDOW_NORMAL);
		imshow("配准结果", vis_align);
	}
	
	// 双光融合
	Mat dst;

	// 基于小波变换的融合方法
	//fusion(vis_align, Infrared, dst);

	// 基于加权求和的融合方法
	//dst = add_fusion(vis_align, Infrared);
	
	// 基于TIF的融合算法
	//TIF(vis_align, Infrared, dst);

	// 基于IEFVIP的融合算法
	//// 设置参数
	//int QuadNormDim = 416;
	//int QuadMinDim = 32;
	//int GaussScale = 3;
	//double MaxRatio = 0.001;
	//double StdRatio = 0.5;
	//dst = BGR_Fuse(vis_align, Infrared, QuadNormDim, QuadMinDim, GaussScale, MaxRatio, StdRatio);

	// 基于GFF的融合算法
	dst = GFF_GRAY(vis_align, Infrared);

	cv::resize(dst, dst, Size(WIDTH, HEIGHT));

	if (dst.data) 
	{
		cv::namedWindow("融合结果", WINDOW_NORMAL);
		imshow("融合结果", dst);
	}

	waitKey();
	destroyAllWindows();

	return 0;
}