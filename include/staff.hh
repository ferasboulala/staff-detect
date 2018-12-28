#ifndef STAFF_H_
#define STAFF_H_

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * \brief Gradient from each horizontal pixel in the staff model.
*/
typedef std::vector<double> Gradient;

/**
 * \brief StaffModel that holds the gradient (or orientation) at each column
 */
typedef struct {
  Gradient gradient;
  int start_col, start_row, staff_height, staff_space;
  double rot;
  bool straight;
  cv::Mat staff_image;
} StaffModel;

/**
 * \brief Vector of staffs with first and last line. Combined with a staff
 * model, it can serve pitch inference
 */
typedef std::vector<std::pair<int, int>> Staffs;

/**
 * \fn StaffModel GetStaffModel(const cv::Mat &src, const int n_threads = 1)
 * \brief Estimates a staff model from a binary image
 * \param src Binary image (CV_8UC1, black on white)
 * \return Estimated model
 */
StaffModel GetStaffModel(const cv::Mat &src, const int n_threads = 1);

/**
 * \fn void PrintStaffModel(cv::Mat &dst, const StaffModel &model)
 * \brief Prints a staff model on a black image
 * \param dst Image on which it will be printed (becomes black)
 */
void PrintStaffModel(cv::Mat &dst, const StaffModel &model);

/**
 * \fn Staffs FitStaffModel(const StaffModel &model)
 * \brief Fits the given model and returns all valid staffs
 * \param StaffModel
 * \return All valid staffs
 */
Staffs FitStaffModel(const StaffModel &model);

/**
 * \fn void PrintStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model)
 * \brief Prints all the detected staffs of the model on an image
 * \param dst Image on which it will be printed (will be BGR)
 * \param model The relevant staff model
 * \return The position of the staffs
 */
void PrintStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model);

/**
 * \fn void RemoveStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model)
 * \brief Removes staffs from an image according to a model and start and end
 * positions
 * \param dst Image with staffs
 * \param staffs The positions of the staffs
 * \param The relevant staff model
 */
void RemoveStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model);

/**
 * \fn void Realign(cv::Mat &dst, const StaffModel &model)
 * \brief Realigns the image according to the staff model gradient. The output
 * image is then modeled with a constant gradient. To distort back the image,
 * give the inverse of the model (*-1).
 * \param Destination image (grayscale)
 * \param model The relevant staff model
 */
void Realign(cv::Mat &dst, const StaffModel &model);

/**
 * \fn void SaveToDisk(const std::string &fn, const Staffs &staffs,
                const StaffModel &model)
 * \brief Saves the staff information in a xml format
 * \param fn Filename
 * \param staffs Staffs
 * \param model The relevant staff model
 */
void SaveToDisk(const std::string &fn, const Staffs &staffs,
                const StaffModel &model);

#endif // STAFF_H_