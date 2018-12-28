#include <util.hh>

void to_binary(const cv::Mat &src, cv::Mat &dst)
{
}

bool is_gray(const cv::Mat &src)
{
  unsigned char depth = src.type() & CV_MAT_DEPTH_MASK;
  return depth == CV_8U && src.channels() == 1;
}

bool is_black_on_white(const cv::Mat &src)
{
  assert(is_gray(src));
  int black = 0, white = 0;
  for (int i = 0; i < src.rows / 2; i++)
  {
    for (int j = 0; j < src.cols / 2; j++)
    {
      if (src.at<char>(i, j) == 0)
        black++;
      else
        white++;
    }
  }
  return (black < white);
}

void bounding_box(const cv::Mat &src, cv::Mat &dst)
{
  cv::Mat points;
  cv::findNonZero(src, points);
  cv::Rect bbox = cv::boundingRect(points);
  dst = dst(bbox);
}

void rotate(const cv::Mat &src, cv::Mat &dst, const double rotation)
{
  if (abs(rotation) < 1)
    return;
  cv::Point2f center((dst.cols - 1) / 2.0, (dst.rows - 1) / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, rotation, 1.0);
  cv::Rect bbox =
      cv::RotatedRect(cv::Point2f(), dst.size(), rotation).boundingRect();
  rot.at<double>(0, 2) += bbox.width / 2.0 - dst.cols / 2.0;
  rot.at<double>(1, 2) += bbox.height / 2.0 - dst.rows / 2.0;
  cv::warpAffine(src, dst, rot, bbox.size());
}

void raw_rotate(const cv::Mat &src, cv::Mat &dst, const double rotation)
{
  if (abs(rotation) < 1)
    return;
  cv::Point2f center((dst.cols - 1) / 2.0, (dst.rows - 1) / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, rotation, 1.0);
  cv::warpAffine(dst, dst, rot, dst.size());
}

std::string strip_fn(const std::string &fn)
{
  const int size = fn.size();
  std::string out = fn;
  for (int i = size - 1; i >= 0; i--)
  {
    if (fn[i] == '/')
    {
      out.clear();
      for (int j = i + 1; j < size; j++)
      {
        out.push_back(fn[j]);
      }
      break;
    }
  }
  return out;
}

std::string strip_ext(const std::string &fn)
{
  const int size = fn.size();
  std::string out = fn;
  for (int i = size - 1; i >= 0; i--)
  {
    if (fn[i] == '.')
    {
      out.clear();
      for (int j = 0; j < i; j++)
      {
        out.push_back(fn[j]);
      }
      break;
    }
  }
  return out;
}