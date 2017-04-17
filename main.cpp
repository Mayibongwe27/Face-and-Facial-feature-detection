#include "detectfeatures.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(){

	DetectFeatures facial_features =  DetectFeatures();
	string input_image_path = ".\\images\\*.JPG"; // Path to training data set
	string input_test_path = ".\\Faces-Test\\*.JPG"; //Path to testing data set
	string face_cascade_path = ".\\cascades\\lbpcascade_frontalface.xml"; //Path to Haar-cascade classifier face
	string eye_cascade_path = ".\\cascades\\haarcascade_eye.xml"; //Path to eye cascade classifier
	string nose_cascade_path = ".\\cascades\\haarcascade_mcs_nose.xml"; //Path to nose cascade classifier 
	string mouth_cascade_path = ".\\cascades\\haarcascade_mcs_mouth.xml"; //Path to mouth cascade classifier


	vector<String> filenames; //hold individual URLs to images
	glob(input_image_path, filenames);
Mat image;
Mat dest;
string label;
int start = input_image_path.find("*");
for (size_t i = 0; i < filenames.size(); i++)
{


	label = filenames[i].substr(start, 6);
	if (input_image_path.empty() || face_cascade_path.empty())
	{
		std::cout << "IMAGE or FACE_CASCADE are not specified";
		return 1;
	}
	// Load image and cascade classifier files
	image = imread(filenames[i]);
	//image = imread(input_image_path);
	if (!image.data)
	{
		std::cout << "Image not read\n";
		//waitKey(10000);
	}

	cvtColor(image, image, CV_BGR2GRAY);
	image.convertTo(image, CV_8UC1);
	GaussianBlur(image, dest, Size(7, 7), 5, 3, BORDER_CONSTANT); //tiny bit of smoothing not a bad idea

	//Show the loaded image
	namedWindow("Show", WINDOW_AUTOSIZE);
	imshow("Show", image);
	waitKey(1000);

	// Detect faces and facial features
	vector<Rect_<int> > faces;

	facial_features.detectFaces(dest, faces, face_cascade_path);

	//std::cout << label << " Face Detected" << endl;
	waitKey(1000);
	facial_features.detectFacialFeatures(dest, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path,label);

	namedWindow("Result", WINDOW_AUTOSIZE);
	imshow("Result", dest);
	waitKey(2000);
}

return 0;
}