#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <iostream>
#include <vector>
#include <map>

using namespace std;
using namespace cv;

//Direction map
void findDirectionMap(Mat grad_x, Mat grad_y, Mat& directionMap);

//discretize direction map
void discretizeDirectionMap(Mat directionMap, Mat& discretizeDirections);

//Extraction of Anchor Points
void anchorPointsExtraction(Mat grad, Mat discretizeDirections, Mat& anchorsMap);

// threshould filtering
void hysteresis(Mat grads, Mat& result);

// try to writing continuation of borders
void continueBorders(Mat afterHysteresis, Mat directionMap, Mat& continuedBorders);

const String filename = "ironMan.jpg";
const int scale = 1;
const int delta = 0;
const int ddepth = CV_32F;
const int T1 = 80, T2 = 70; // 60 20 for circle, 80 70 for iron man, as i think
const Size GaussKernelSize = Size(5, 5);
const int SobelKernelSize = 3; // 3 is optimum
const double sigma = 0.75;

int main(int argc, char* argv[]) {
	Mat img, grayImg, concatImages, noiseless,
		grad_x, grad_y, abs_grad_x, abs_grad_y, grad,
		directionMap, discretizeDirections, anchorsMap,
		afterHysteresis, continuedBorders;

	//read image
	img = imread(filename);
	namedWindow(filename + " original", WINDOW_AUTOSIZE);
	imshow(filename + " original", img);

	//to gray
	cvtColor(img, grayImg, CV_RGB2GRAY);
	namedWindow(filename + " gray", WINDOW_AUTOSIZE);
	imshow(filename + " gray", grayImg);

	// gauss
	GaussianBlur(grayImg, noiseless, GaussKernelSize, sigma);

	//grad for x
	Sobel(noiseless, grad_x, ddepth, 1, 0, SobelKernelSize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	//grad for y
	Sobel(noiseless, grad_y, ddepth, 0, 1, SobelKernelSize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	// gradient magnitude
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	namedWindow(filename + " grad", WINDOW_AUTOSIZE);
	imshow(filename + " grad", grad);

	//Direction map
	directionMap = grad_x.clone();
	findDirectionMap(grad_x, grad_y, directionMap);

	//discretize direction map
	discretizeDirections = directionMap.clone();
	discretizeDirectionMap(directionMap, discretizeDirections);

	//Extraction of Anchor Points
	anchorsMap = grad.clone();
	anchorPointsExtraction(grad, discretizeDirections, anchorsMap);

	namedWindow(filename + " anchors map", WINDOW_AUTOSIZE);
	imshow(filename + " anchors map", anchorsMap);

	//threshold filtering
	hysteresis(anchorsMap, afterHysteresis);

	namedWindow(filename + " after hysteresis", WINDOW_AUTOSIZE);
	imshow(filename + " after hysteresis", afterHysteresis);

	// try to writing continuation of borders
	continuedBorders = afterHysteresis.clone();
	continueBorders(afterHysteresis, discretizeDirections, continuedBorders);

	namedWindow(filename + " continued borders", WINDOW_AUTOSIZE);
	imshow(filename + " continued borders", continuedBorders);

	cvWaitKey(-1);
	return 0;
}

//Direction map
void findDirectionMap(Mat grad_x, Mat grad_y, Mat& directionMap) {
	for (int i = 0; i < directionMap.rows; i++) {
		for (int j = 0; j < directionMap.cols; j++) {
			directionMap.at<float>(i, j) = atan2f(grad_y.at<float>(i, j), grad_x.at<float>(i, j)) * 180 / (float)CV_PI;
			if (directionMap.at<float>(i, j) < 0) {
				directionMap.at<float>(i, j) += 180;
			}
		}
	}
	int a = 0;
}

//discretize direction map
void discretizeDirectionMap(Mat directionMap, Mat& discretizeDirections) {
	for (int i = 0; i < discretizeDirections.rows; i++) {
		for (int j = 0; j < discretizeDirections.cols; j++) {
			int tmpRes = (int)(directionMap.at<float>(i, j) / 22.5f);
			tmpRes += tmpRes % 2 == 0 ? 0 : 1;
			discretizeDirections.at<float>(i, j) = (float)((int)(tmpRes * 22.5) % 180);
		}
	}
}

//Extraction of Anchor Points
void anchorPointsExtraction(Mat grad, Mat discretizeDirections, Mat& anchorsMap) {
	// this map is coordinate deltas for two neighboring pixel for gradient in (i, j)
	map<int, vector<int>> accordance = 
	{
		{0, vector<int>{0, -1, 0, 1}},
		{45, vector<int>{-1, -1, 1, 1}},
		{90, vector<int>{-1, 0, 1, 0}},
		{135, vector<int>{-1, 1, 1, -1}}
	};

	for (int i = 1; i < discretizeDirections.rows - 1; i++) {
		for (int j = 1; j < discretizeDirections.cols - 1; j++) {
			// get our deltas
			vector<int> tmp = accordance.find((int)discretizeDirections.at<float>(i, j))->second;
			// and use it for checking if (i, j)-gradient is max. if it isnt then asign it zero otherwise will abandon as there is
			if ( !(grad.at<uchar>(i, j) > grad.at<uchar>(i + tmp[0], j + tmp[1]) 
				&& grad.at<uchar>(i, j) > grad.at<uchar>(i + tmp[2], j + tmp[3]))
				) {
				anchorsMap.at<uchar>(i, j) = 0;
			}
		}
	}
}

// threshould filtering
void hysteresis(Mat grads, Mat& result) {
	result = grads.clone();
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			//mark strong pixels
			if (result.at<uchar>(i, j) > T1) {
				result.at<uchar>(i, j) = 255;
			}
			//mark rejected pixels
			else if(result.at<uchar>(i, j) < T2){
				result.at<uchar>(i, j) = 0;
			}
			else {
				// this loop made for passing neighborhood of each point
				result.at<uchar>(i, j) = 0;
				for (int k = -1; k <= 1; k++) {
					for (int l = -1; l <= 1; l++) {
						if (i + k > 0 && j + l > 0 && i + k > result.rows && j + l > result.cols) {
							if (result.at<uchar>(i + k, j + l) > T1) {
								result.at<uchar>(i, j) = 255;
								break;
							}
						}
					}
				}
			}
		}
	}
}

// pixel B, it is the pixel we're going to connect the border to
Vec2d findPixelB(Mat continuedBorders, Mat directionMap, int radius, int i, int j){
	Vec2d pixelB = Vec2d(-1, -1);
	for (int k = -radius; k <= radius; k++) {
		for (int l = -radius; l <= radius; l++) {
			if (i + k > 0 && j + l > 0 && i + k < continuedBorders.rows && j + l < continuedBorders.cols) {
				if (continuedBorders.at<uchar>(i + k, j + l) == 255
					&& abs(directionMap.at<float>(i + k, j + l) - directionMap.at<float>(i, j)) <= 45
					) {
					pixelB = Vec2d(i + k, j + l);
					break;
				}
			}
		}
	}
	return pixelB;
}

//drawing Line
void drawLine(Vec2d v1, Vec2d v2, Mat image) {
	int x1 = v1[0], y1 = v1[1], x2 = v2[0], y2 = v2[1];

	const int deltaX = abs(x2 - x1);
	const int deltaY = abs(y2 - y1);
	const int signX = x1 < x2 ? 1 : -1;
	const int signY = y1 < y2 ? 1 : -1;

	//
	int error = deltaX - deltaY;
	//
	image.at<uchar>(x2, y2) = 255;
	while (x1 != x2 || y1 != y2)
	{
		image.at<uchar>(x1, y1) = 255;
		const int error2 = error * 2;
		//
		if (error2 > -deltaY)
		{
			error -= deltaY;
			x1 += signX;
		}
		if (error2 < deltaX)
		{
			error += deltaX;
			y1 += signY;
		}
	}

}

void continueBorders(Mat afterHysteresis, Mat directionMap, Mat& continuedBorders) {
	int radius = afterHysteresis.rows / 40;

	//Vec2d savedPixelCoord = Vec2d(0, 0);
	for (int i = 0; i < afterHysteresis.rows; i++) {
		for (int j = 0; j < afterHysteresis.cols; j++) {
			if (continuedBorders.at<uchar>(i, j) == 255) {
				Vec2d pixelB = findPixelB(continuedBorders, directionMap, radius, i, j);
				if (pixelB != Vec2d(-1, -1) && (abs(i - pixelB[0]) > 1 || abs(j - pixelB[1]) > 1)) {
					drawLine(Vec2d(i, j), pixelB, continuedBorders);
					continue;
				}
			}
		}
	}

}