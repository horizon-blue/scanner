#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


// Reference:
// https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

auto find_squares(Mat& image) {
    vector<vector<Point>> squares;

    // blur will enhance edge detection
    Mat blurred(image);
    medianBlur(image, blurred, 9);

    Mat gray0(blurred.size(), CV_8U), gray;
    vector<vector<Point>> contours;

    // find squares in every color plane of the image
    for(int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&blurred, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        const int threshold_level = 2;
        for(int l = 0; l < threshold_level; l++) {
            // Use Canny instead of zero threshold level!
            // Canny helps to catch squares with gradient shading
            if(l == 0) {
                Canny(gray0, gray, 10, 20, 3);  //

                // Dilate helps to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
            } else {
                gray = gray0 >= (l + 1) * 255 / threshold_level;
            }

            // Find contours and store them in a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            // Test contours
            vector<Point> approx;
            for(size_t i = 0; i < contours.size(); i++) {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]),
                             approx,
                             arcLength(Mat(contours[i]), true) * 0.02,
                             true);

                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if(approx.size() == 4 &&
                   fabs(contourArea(Mat(approx))) > 1000 &&
                   isContourConvex(Mat(approx))) {
                    double maxCosine = 0;

                    for(int j = 2; j < 5; j++) {
                        double cosine = fabs(
                            angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    if(maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }

    return squares;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << " Usage: scanner <image_to_scan>" << endl;
        return -1;
    }

    auto img = imread(argv[1], CV_LOAD_IMAGE_COLOR);  // Read the file
    // imshow("Display window", img);
    // waitKey(0);

    auto squares = find_squares(img);


    // Plot the outline of box
    cout << "Found " << squares.size() << " box(es)" << endl;
    for(auto& square : squares) {
        for(int i = 0; i < 4; ++i) {
            line(img,
                 square[i % 4],
                 square[(i + 1) % 4],
                 CV_RGB(150, 200, 235),
                 10);
        }
    }

    imwrite("./out.jpg", img);

    return 0;
}