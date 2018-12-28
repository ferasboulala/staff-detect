#ifndef _UTIL_HH
#define _UTIL_HH

#include <string>
#include <opencv2/opencv.hpp>

/**
 * \fn void to_binary(const cv::Mat &src)
 * \brief Converts an input image to a binary image.
 * \param src Input image.
 * \param dst Output image
*/
void to_binary(const cv::Mat &src, cv::Mat &dst);

/**
 * \fn bool is_gray(const cv::Mat &src)
 * \brief Checks if an image has a gray type.
 * \param src Input image.
 * \return Whether or not the image is grayscale.
*/
bool is_gray(const cv::Mat &src);

/**
 * \fn bool is_black_on_white(const cv::Mat &src)
 * \brief Checks if an image is black on white.
 * \param src Input image.
 * \return Whether or not the image is black on white.
*/
bool is_black_on_white(const cv::Mat &src);

/**
 * \fn void bounding_box(const cv::Mat &src, cv::Mat &dst)
 * \brief Gets the ROI bouding box on an input image.
 * \param src Input image.
 * \param dst Output image.
*/
void bounding_box(const cv::Mat &src, cv::Mat &dst);

/**
 * \fn void rotate(const cv::Mat &src, cv::Mat &dst, const double rotation)
 * \brief Rotates an input image with a given rotation in degrees clockwise.
 * It will crop the result so only the original pixels remain.
 * \param src Input image.
 * \param dst Output image.
 * \param rotation Rotation in degrees.
*/
void rotate(const cv::Mat &src, cv::Mat &dst, const double rotation);

/**
 * \fn void raw_rotate(cv::Mat &dst, const double rot_theta)
 * \brief Rotates an input image with a given rotation in degrees clockwise.
 * Black borders will appear.
 * \param src Input image.
 * \param dst Output image.
 * \param rotation Rotation in degrees.
*/
void raw_rotate(const cv::Mat &src, cv::Mat &dst, const double rotation);

/**
 * \fn std::string strip_fn(const std::string &fn)
 * \brief Strips the path of a filename
 * \param fn Absolute or relative filename
 * \return Name of the file
*/
std::string strip_fn(const std::string &fn);

/**
 * \fn std::string strip_ext(const std::string &fn)
 * \brief Strips the extension of a file
 * \param fn Path to the file
 * \return Path without the extension
*/
std::string strip_ext(const std::string &fn);

/**
 * \fn inline bool is_image(const std::string &fn)
 * \brief Checks if the filename is an image supported by OpenCV
 * \param fn The filename
 * \return Whether or not the filename is a valid image
*/
inline bool is_image(const std::string &fn){
  return !(fn.find(".png") == std::string::npos &&
        fn.find(".jpg") == std::string::npos &&
        fn.find(".PNG") == std::string::npos);
}

#endif // _UTIL_HH