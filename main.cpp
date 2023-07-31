#include <string>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const int slider_max = 100;
int temperature_amount = 50;
int tint_amount = 50;
int brightness_amount = 50;
int contrast_amount = 50;
int sharpen_amount = 50;
int vignette_amount = 50;

const std::string window_name = "Image Modifier";

cv::Mat ori, dst;

struct Adjustments {
  double temperature = 0.f; 
  double tint = 0.f;        
  double bright = 0.f;      
  double contrast = 0.f;    
  double sharpen = 0.f;     
  double vignette = 0.f;    
} adjustments;

void ChangeTemperature(cv::Mat in, cv::Mat out, float adjust) {
  // TODO
  in.copyTo(out);
  vector<double> channel_R(256), channel_B(256);
  adjust = (adjust + 1) * 50;
  // cout << "===========" << adjust << "==============" << endl;
  for (int i = 0; i < 256; i++)
  {
      channel_R[i] = cv::saturate_cast<uchar>(i + (adjust - 50) / 2);
      channel_B[i] = cv::saturate_cast<uchar>(i - (adjust - 50) / 2);
  }
  vector<double> channel_G(channel_R);
  size_t n_pixel = in.total();
  auto p_out = out.ptr<cv::Vec3b>();
  auto p_in = in.ptr<cv::Vec3b>();
  for (size_t i = 0; i < n_pixel; i++)
  {
      p_out[i][0] = channel_B[p_in[i][0]];
      p_out[i][1] = channel_G[p_in[i][1]];
      p_out[i][2] = channel_R[p_in[i][2]];
  }
}

void ChangeTint(cv::Mat in, cv::Mat out, float adjust) {
    // TODO
    in.copyTo(out);
    vector<double> channel_R(256), channel_G(256);
    adjust = (adjust + 1) * 50;
    // cout << "===========" << adjust << "==============" << endl;
    for (int i = 0; i < 256; i++)
    {
        channel_R[i] = cv::saturate_cast<uchar>(i + (adjust - 50) / 2);
        channel_G[i] = cv::saturate_cast<uchar>(i - (adjust - 50) / 2);
    }
    vector<double> channel_B(channel_R);
    size_t n_pixel = in.total();
    auto p_out = out.ptr<cv::Vec3b>();
    auto p_in = in.ptr<cv::Vec3b>();
    for (size_t i = 0; i < n_pixel; i++)
    {
        p_out[i][0] = channel_B[p_in[i][0]];
        p_out[i][1] = channel_G[p_in[i][1]];
        p_out[i][2] = channel_R[p_in[i][2]];
    }
}

void ChangeBrightness(cv::Mat in, cv::Mat out, float adjust) {
  double gamma = abs(adjust) * 9 + 1;
  ;
  if (adjust > 0) {
    gamma = 1 / gamma;
  }
  std::vector<uchar> curve(256);
  for (int i = 0; i < 256; ++i) {
    float a = i / 255.f;
    float b = pow(a, gamma);
    curve[i] = cv::saturate_cast<uchar>(b * 255.f + 0.5f);
  }
  size_t n_pixel = in.total();
  auto p_out = out.ptr<cv::Vec3b>();
  auto p_in = in.ptr<cv::Vec3b>();
  for (size_t i = 0; i < n_pixel; ++i) {
    for (int c = 0; c < ori.channels(); ++c) {
      p_out[i][c] = curve[p_in[i][c]];
    }
  }
}

void ChangeContrast(cv::Mat in, cv::Mat out, float adjust) {
  // TODO: implement it
  in.copyTo(out);
  vector<uchar> curve(256);
  for (int i = 0; i < 256; i++)
  {
      curve[i] = cv::saturate_cast<uchar>(adjust * (i - 127) + 127);
  }
  size_t n_pixel = in.total();
  auto p_out = out.ptr<cv::Vec3b>();
  auto p_in = in.ptr<cv::Vec3b>();
  for (size_t i = 0; i < n_pixel; ++i) {
      for (int c = 0; c < ori.channels(); ++c) {
          p_out[i][c] = curve[p_in[i][c]];
      }
  }
}

void Sharpen(cv::Mat in, cv::Mat out, float adjust) {
  // TODO: implement it
  in.copyTo(out);
  cv::Mat temp(in);
  Mat fliter = (Mat_<float>(3, 3) << -adjust, -adjust, -adjust, -adjust, 8 * adjust + 1, -adjust, -adjust, -adjust, -adjust);
  filter2D(in, out, CV_8UC3, fliter);

  //too slow
  /*for (int i = 1; i < (in.rows - 1); i++)
  {
      for (int j = 1; j < (in.cols - 1); j++)
      {

          for (int k = 0; k < 3; k++)
          {
              out.at<Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(in.at<Vec3b>(i - 1, j - 1)[k] * filter_four_area[0] +
                  in.at<Vec3b>(i - 1, j)[k] * filter_four_area[1] + in.at<Vec3b>(i - 1, j + 1)[k] * filter_four_area[2] +
                  in.at<Vec3b>(i, j - 1)[k] * filter_four_area[3] + in.at<Vec3b>(i, j)[k] * filter_four_area[4] +
                  in.at<Vec3b>(i, j + 1)[k] * filter_four_area[5] + in.at<Vec3b>(i + 1, j - 1)[k] * filter_four_area[6] +
                  in.at<Vec3b>(i + 1, j)[k] * filter_four_area[7] + in.at<Vec3b>(i + 1, j + 1)[k] * filter_four_area[8]);
          }
      }
  }*/
}

void AddVignette(cv::Mat in, cv::Mat out, float adjust) {
  in.copyTo(out);

  Mat result = in.clone();
  result.convertTo(result, CV_32FC1);
  int width = in.cols;
  int height = in.rows;
  int centerX = width / 2;
  int centerY = height / 2;
  for (int i = 0; i < height; i++)
  {
      for (int j = 0; j < width; j++)
      {
          double distance = sqrt((j - centerX) * (j - centerX) + (i - centerY) * (i - centerY));
          double fix = sqrt(distance) * adjust;
          for (int k = 0; k < 3; k++) {
              if (fix > 255) fix = 255;
              if (result.at<Vec3f>(i, j)[k] + fix > 255) result.at<Vec3f>(i, j)[k] = 255;
              else result.at<Vec3f>(i, j)[k] += fix;
          }
      }
  }
  result.convertTo(out, CV_8UC1);
}

void ProcessAndShow() {
  ChangeTemperature(ori, dst, adjustments.temperature);
  ChangeTint(dst, dst, adjustments.tint);
  ChangeBrightness(dst, dst, adjustments.bright);
  ChangeContrast(dst, dst, adjustments.contrast);
  Sharpen(dst, dst, adjustments.sharpen);
  AddVignette(dst, dst, adjustments.vignette);
  cv::imshow(window_name, dst);
}

static void on_brightness(int, void *) {
  adjustments.bright =
      (static_cast<double>(brightness_amount) / slider_max - 0.5f) *
      2;  // (-1, 1)
  ProcessAndShow();
}

static void on_contrast(int, void *) {
  adjustments.contrast =
      (static_cast<double>(contrast_amount) / slider_max) *
      2;  // (0, 2)
  ProcessAndShow();
}

static void on_temperature(int, void *) {
  adjustments.temperature =
      (static_cast<double>(temperature_amount) / slider_max - 0.5f) *
      2;  // (-1, 1)
  ProcessAndShow();
}

static void on_sharpen(int, void*) {
    adjustments.sharpen = 
        (static_cast<double>(sharpen_amount) / slider_max - 0.5f) * 2; //(-1, 1)
        //(static_cast<double>(0.5f - sharpen_amount) / slider_max) * 2 + 10;  // (8, 10)
    ProcessAndShow();
}

static void on_vignette(int, void*) {
    adjustments.vignette =
        (static_cast<double>(vignette_amount) / slider_max);  // (0, 2)
    ProcessAndShow();
}

static void on_tint(int, void*) {
    adjustments.tint =
        (static_cast<double>(tint_amount) / slider_max - 0.5f) * 2;  // (-1, 1)
    ProcessAndShow();
}

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, "{@input|E:/image/Torly.jpg|input image}");
  // Read images
  ori = cv::imread(parser.get<std::string>("@input"));
  if (ori.empty()) {
    std::cout << "failed to load image, check input path.\n";
    return -1;
  }
  // Resize for easier use, you can change this behaviour.
  cv::Size new_size = cv::Size(800, 600);
  cv::resize(ori, ori, new_size);
  dst.create(ori.size(), ori.type());

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  // Add trackbar
  cv::createTrackbar("Tint", window_name, &tint_amount, 100,
      on_tint);
  cv::createTrackbar("Brightness", window_name, &brightness_amount, 100,
                     on_brightness);
  cv::createTrackbar("Contrast", window_name, &contrast_amount, 100,
                     on_contrast);
  cv::createTrackbar("Temperature", window_name, &temperature_amount, 100,
                     on_temperature);    
  cv::createTrackbar("Sharpen", window_name, &sharpen_amount, 100,
      on_sharpen);
  cv::createTrackbar("Vignette", window_name, &vignette_amount, 100,
      on_vignette);

  ProcessAndShow();
  cv::waitKey();
}
