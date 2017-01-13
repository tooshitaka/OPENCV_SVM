//#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <opencv2/video/background_segm.hpp>
#include <opencv2\video\tracking.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv::ml;

void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	std::cout << rows << std::endl;
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	std::cout << cols << std::endl;
	std::cout << "#Rows after pop" << train_samples[0].rows << std::endl;
	std::cout << "#Cols after pop" << train_samples[0].cols << std::endl;
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	std::vector< cv::Mat >::const_iterator itr = train_samples.begin();
	std::vector< cv::Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			cv::transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

int main(int argc, char** argv)
{
	// Load input image
	cv::Mat LoadedImage;
	LoadedImage = cv::imread("lena.jpg", 1);
	cv::cvtColor(LoadedImage, LoadedImage, CV_BGR2GRAY);

	std::cout << "Number of width:" << LoadedImage.rows << std::endl;
	std::cout << "Number of height:" << LoadedImage.cols << std::endl;

	// Show the input image
	cv::namedWindow("InputImage", cv::WINDOW_AUTOSIZE);
	cv::imshow("InputImage", LoadedImage);
	cv::waitKey(1000);

	// Parameters for the sliding window
	int windows_n_rows = 80;
	int windows_n_cols = 80;
	// Step of each window
	int StepSlide = 80;

	// Copy Loaded image
	cv::Mat DrawResultGrid = LoadedImage.clone();

	cv::TickMeter meter;
	meter.start();
	std::vector<cv::Mat> img_lst;
	std::vector<int> label;
	int count = 1;
	// Cycle row step
	// Note the end condition: the rectange is drawn by left-top position
	for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
	{
		// Cycle col step
		for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
		{
			// 顔検出とか物体検出とかするときはここに処理を書く

			// resulting window   
			cv::Rect windows(col, row, windows_n_rows, windows_n_cols);

			cv::Mat DrawResultHere = LoadedImage.clone();

			// Draw only rectangle
			cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
			// Draw grid
			cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

			// Selected windows roi
			cv::Mat roi = LoadedImage(windows);
			cv::Mat sample = roi.clone();
			if (count == 1) {
				img_lst.push_back(sample.reshape(0, 1));
				label.push_back(-1);
			}
			else if (count == 2) {
				img_lst.push_back(sample.reshape(0, 1));
				label.push_back(+1);
			}
			//// Show  rectangle
			cv::namedWindow("Step 2 draw Rectangle", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
			cv::imshow("Step 2 draw Rectangle", DrawResultHere);
			cv::imshow("ROI", roi);
			cv::waitKey(100);

			count++;
		}
	}
	meter.stop();
	std::cout << meter.getTimeMilli() << "ms" << std::endl;

	cv::Mat train_data;
	convert_to_ml(img_lst, train_data);
	std::cout << train_data.rows << std::endl;
	std::cout << train_data.cols << std::endl;

	cv::Ptr<cv::ml::TrainData> train = cv::ml::TrainData::create(train_data, SampleTypes::ROW_SAMPLE, cv::Mat(label));
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train);
	//svm->train(train_data, ROW_SAMPLE, Mat(labels));
	if (svm->isTrained())
		std::cout << "Trained!!!" << std::endl;
	for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
	{
		// Cycle col step
		for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
		{
			// 顔検出とか物体検出とかするときはここに処理を書く

			// resulting window   
			cv::Rect windows(col, row, windows_n_rows, windows_n_cols);

			cv::Mat DrawResultHere = LoadedImage.clone();

			// Draw only rectangle
			cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
			// Draw grid
			cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

			// Selected windows roi
			cv::Mat roi = LoadedImage(windows);
			cv::Mat sample = roi.clone();
			cv::Mat test = sample.reshape(0, 1);
			test.convertTo(test, CV_32F);
			std::cout << svm->predict(test) << std::endl;
			//// Show  rectangle
			cv::namedWindow("Step 2 draw Rectangle", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
			cv::imshow("Step 2 draw Rectangle", DrawResultHere);
			cv::imshow("ROI", roi);
			cv::waitKey(100);

			count++;
		}
	}
	return 0;
}