// Local includes
#include <staff.hh>
#include <util.hh>
#include <tinyxml2.hh>

// c++ std
#include <thread>

// c std
#include <assert.h>
#include <unistd.h>

// Hyperparameters bellow

#define BINARY_THRESH_VAL 128
// Minimum amount of CC in a column to estimate the gradient
#define MIN_CONNECTED_COMP 10
// Number of neighbouring CC to average the gradient
#define K_NEAREST 5
// Size of the sliding window in number of lines to find staffs
#define KERNEL_SIZE 5
// Ratio of the max amount of polls per line to consider it a valid line
#define MIN_POLL_PER_LINE_RATIO 0.75
// Ratio of the max amount of polls per staff to suspect the presence of a one
#define POLL_PER_STAFF_RATIO 0.6
// Minimum amount of detected HoughLines to consider it straight
#define MIN_HOUGH_LINES 10

// Hough
#define THETA_RES 2
#define N_BINS 20

// Useful constants
#define LINES_PER_STAFF 5
#define RAD2DEG 180 / CV_PI
#define DEG2RAD 1 / (RAD2DEG)
#define EPSILON 1e-7

namespace
{

void draw_model(cv::Mat &dst, const StaffModel &model, const int pos,
                const cv::Scalar color)
{
  assert(!is_gray(dst));
  double y = pos;
  for (auto g = model.gradient.begin(); g != model.gradient.end(); g++)
  {
    y += *g;
    const int x = g - model.gradient.begin() + model.start_col;
    if ((y > dst.rows) || (y < 0))
      continue;
    dst.at<cv::Vec3b>((int)round(y), x)[0] = color[0];
    dst.at<cv::Vec3b>((int)round(y), x)[1] = color[1];
    dst.at<cv::Vec3b>((int)round(y), x)[2] = color[2];
  }
}

void estimate_rotation(cv::Mat &img, StaffModel &model)
{
  typedef std::vector<cv::Vec2f> Lines;
  Lines lines;
  cv::HoughLines(img, lines, 1, CV_PI / (180 * THETA_RES), img.cols / 2);

  // Storing them into a histogram
  std::vector<Lines> histogram(N_BINS, Lines());
  for (auto line : lines)
  {
    const double theta = line[1] * RAD2DEG;
    const int hist_idx = (int)theta / (int)(CV_PI * RAD2DEG / N_BINS);
    histogram[hist_idx].push_back(line);
  }

  // Getting the most populated bin from the histogram
  Lines maximum_polled_lines;
  for (auto &bin : histogram)
    if (bin.size() > maximum_polled_lines.size())
      maximum_polled_lines = bin;

  // Getting the average of that bin
  double avg_theta = 0;
  int diag = sqrt(pow(img.cols, 2) + pow(img.rows, 2));
  for (auto line : maximum_polled_lines)
  {
    const double theta = line[1];
    avg_theta += theta;
  }
  avg_theta /= (maximum_polled_lines.size() + EPSILON);

  // If 70% of the lines are in it and there are at least more than 10 lines,
  // use a constant model (flat gradient)
  model.rot = CV_PI / 2;
  model.straight = false;
  if ((double)maximum_polled_lines.size() / lines.size() >= 0.7 &&
      lines.size() > MIN_HOUGH_LINES)
  {
    rotate(img, img, RAD2DEG * (avg_theta - CV_PI / 2));
    model.straight = true;
    model.rot = avg_theta;
  }
  cv::Mat points;
  cv::findNonZero(img, points);
  cv::Rect bbox = cv::boundingRect(points);
  img = img(bbox);
  model.start_col = bbox.x;
  model.start_row = bbox.y;
}

struct RunLengthData
{
  cv::Mat img;
  std::vector<int> *whites, *blacks;
  int start, finish;
};

void run_length_p(struct RunLengthData &data)
{
  const cv::Mat img = data.img;
  for (int x = data.start; x < data.finish; x++)
  {
    int val = img.at<char>(0, x);
    int count = 1;
    for (int y = 1; y < img.rows; y++)
    {
      if (!((val == 0) == (img.at<char>(y, x) == 0)))
      {
        if (!val)
          (*data.blacks)[count]++;
        else
          (*data.whites)[count]++;
        val = img.at<char>(y, x);
        count = 1;
      }
      else
        count++;
    }
  }
}

void run_length(const cv::Mat &img, int &staff_height, int &staff_space,
                const int n_threads)
{
  std::vector<std::vector<int>> white_run_length(n_threads,
                                                 std::vector<int>(img.rows, 0));
  std::vector<std::vector<int>> black_run_length(n_threads,
                                                 std::vector<int>(img.rows, 0));
  std::vector<std::thread> threads(n_threads);
  std::vector<struct RunLengthData> run_length_data(n_threads);

  const int cols_per_thread = img.cols / n_threads;
  for (int i = 0; i < n_threads; i++)
  {
    const int start = i * cols_per_thread;
    int end = (i + 1) * cols_per_thread;
    if (i == n_threads - 1)
      end = img.cols;
    run_length_data[i].img = img;
    run_length_data[i].start = start;
    run_length_data[i].finish = end;
    run_length_data[i].whites = &white_run_length[i];
    run_length_data[i].blacks = &black_run_length[i];
    threads[i] = std::thread(run_length_p, std::ref(run_length_data[i]));
  }
  for (auto it = threads.begin(); it != threads.end(); it++)
  {
    it->join();
  }

  // staff_height and staff_space are assigned to the most polled runs
  std::vector<int> white_poll(img.rows, 0), black_poll(img.rows, 0);
  for (int i = 0; i < img.rows; i++)
  {
    for (int j = 0; j < n_threads; j++)
    {
      white_poll[i] += white_run_length[j][i];
      black_poll[i] += black_run_length[j][i];
    }
  }
  int max_polled = 0;
  for (auto it = white_poll.begin(); it != white_poll.end(); it++)
  {
    if (max_polled < *it)
    {
      max_polled = *it;
      staff_height = it - white_poll.begin();
    }
  }
  max_polled = 0;
  for (auto it = black_poll.begin(); it != black_poll.end(); it++)
  {
    if (max_polled < *it)
    {
      max_polled = *it;
      staff_space = it - black_poll.begin();
    }
  }
}

void remove_glyphs(cv::Mat &staff_image, const int staff_height,
                   const int staff_space)
{
  // If you do img.copyTo(staff_image), you get fucked up errors
  const int T = staff_height + 1;
  for (int x = 0; x < staff_image.cols; x++)
  {
    int val = staff_image.at<char>(0, x);
    int count = 1;
    for (int y = 1; y < staff_image.rows; y++)
    {
      if ((val == 0) != (staff_image.at<char>(y, x) == 0))
      {
        if (val != 0)
        {
          if (count > T)
          {
            for (int k = y - 1; k >= y - count - 1; k--)
              staff_image.at<char>(k, x) = 0;
          }
        }
        count = 1;
        val = staff_image.at<char>(y, x);
      }
      else
        count++;
    }
  }
}

struct ConnectedComponent
{
  int n, x, y;
};

struct GradientData
{
  cv::Mat staff_image;
  int start, end;
  std::vector<std::vector<struct ConnectedComponent>> *components;
};

void estimate_gradient_p(struct GradientData &data)
{
  for (int x = data.start; x < data.end; x++)
  {
    int count = 1;
    int val = data.staff_image.at<char>(0, x);
    for (int y = 1; y < data.staff_image.rows; y++)
    {
      if ((val == 0) != (data.staff_image.at<char>(y, x) == 0) ||
          y == data.staff_image.rows - 1)
      {
        if (val != 0)
        {
          struct ConnectedComponent cc;
          cc.x = x;
          cc.y = y - 1;
          cc.n = count;
          (*data.components)[x].push_back(cc);
        }
        val = data.staff_image.at<char>(y, x);
        count = 1;
      }
      else
        count++;
    }
  }
}

void estimate_gradient(StaffModel &model, const int n_threads)
{
  // Getting all connected components of each column
  cv::Mat staff_image = model.staff_image;
  std::vector<std::thread> threads(n_threads);
  std::vector<std::vector<struct ConnectedComponent>> components(
      staff_image.cols);

  std::vector<struct GradientData> data(n_threads);
  const int cols_per_thread = staff_image.cols / n_threads;
  for (int i = 0; i < n_threads; i++)
  {
    const int start = i * cols_per_thread;
    int end = (i + 1) * cols_per_thread;
    if (i == n_threads - 1)
    {
      end = staff_image.cols;
    }
    data[i].staff_image = staff_image;
    data[i].start = start;
    data[i].end = end;
    data[i].components = &components;
    threads[i] = std::thread(estimate_gradient_p, std::ref(data[i]));
  }
  for (auto it = threads.begin(); it != threads.end(); it++)
  {
    it->join();
    const int idx = it - threads.begin();
  }

  // Computing the orientation at each column
  std::vector<double> orientations(staff_image.cols, staff_image.rows);
  // For every column
  for (int x = 0; x < staff_image.cols; x++)
  {
    double global_orientation = 0;
    int global_count = 0;
    if (components[x].size() < MIN_CONNECTED_COMP)
      continue;
    // For every connected component in that column
    for (auto cc : components[x])
    {
      double local_orientation = 0;
      int local_count = 0;
      // For every K nearest component
      for (int k = 1; k <= K_NEAREST; k++)
      {
        const int next_idx = k + x;
        if (next_idx >= staff_image.cols)
          break;
        double row_dist = staff_image.rows;
        for (auto next_cc = components[next_idx].begin();
             next_cc != components[next_idx].end(); next_cc++)
        {
          // If we are getting closer
          const int cur_dist =
              round(next_cc->y - (next_cc->n / 2.0)) - (cc.y - (cc.n / 2.0));
          if (abs(row_dist) > abs(cur_dist))
          {
            row_dist = cur_dist;
          }
          else
            break;
        }
        if (abs(row_dist) <= 2 * k)
        { // Not in paper
          local_orientation += row_dist / k;
          local_count++;
        }
      }
      if (local_count)
      {
        global_orientation += local_orientation / local_count;
        global_count++;
      }
    }
    if (global_count)
    {
      orientations[x] = global_orientation / global_count;
    }
  }
  model.gradient = orientations;
}

void interpolate_model(StaffModel &model)
{
  auto &orientations = model.gradient;
  cv::Mat staff_image = model.staff_image;

  // Interpolating empty columns
  double prev_orientation = staff_image.rows,
         next_orientation = staff_image.rows;
  for (int i = 0; i < orientations.size(); i++)
  {
    if (orientations[i] != staff_image.rows)
    {
      prev_orientation = orientations[i];
      continue;
    }
    int current = i;
    while (i < orientations.size())
    {
      if (orientations[i] != staff_image.rows)
      {
        next_orientation = orientations[i];
        break;
      }
      i++;
    }
    // If one orientation is undefined, copy their interpolation (slope = 0)
    if ((prev_orientation == staff_image.rows) &&
        (next_orientation != staff_image.rows))
    {
      prev_orientation = next_orientation;
    }
    else if ((prev_orientation != staff_image.rows) &&
             (next_orientation == staff_image.rows))
    {
      next_orientation = prev_orientation;
    }
    const double delta_slope =
        (next_orientation - prev_orientation) / (i - current);
    for (int j = current; j < i; j++)
    {
      orientations[j] = prev_orientation + (i - j) * delta_slope;
    }
  }
}

std::vector<int> poll_lines(const StaffModel &model)
{
  cv::Mat img = model.staff_image;
  const bool straight = model.straight;

  // Polling each staff line and keep only the ones
  const int n_rows = img.rows;
  int max = 0;
  std::vector<int> staff_lines(n_rows + 1, 0); // last element is the max
  for (int y = 0; y < n_rows; y++)
  {
    int poll = 0;
    double estimated_y = y;
    for (auto it = model.gradient.begin(); it != model.gradient.end(); it++)
    {
      estimated_y += *it;
      const int x = it - model.gradient.begin();
      const int rounded_y = round(estimated_y);
      // Boundary check
      if (estimated_y > n_rows || estimated_y < 0)
        continue;
      // Model fits
      else if (img.at<char>(rounded_y, x))
        poll++;
    }
    staff_lines[y] = poll;
    if (poll > max)
      max = poll;
  }
  staff_lines[staff_lines.size() - 1] = max;
  return staff_lines;
}

void remove_line(cv::Mat &dst, double line_pos, const StaffModel &model)
{
  assert(is_gray(dst));

  for (int j = 0; j < model.gradient.size();
       j++, line_pos += model.gradient[j])
  {
    const int rounded_pos = round(line_pos);
    if ((rounded_pos > dst.rows) || (rounded_pos < 0))
      continue;
    const int col = j + model.start_col;
    // If the pixel is white, check the size of its CC
    if (dst.at<char>(rounded_pos, col))
    {
      int up = 0;
      for (int k = 1; k <= model.staff_space / 2 && rounded_pos - k >= 0; k++)
      {
        if (dst.at<char>(rounded_pos - k, col))
        {
          up++;
        }
        else
        {
          break;
        }
      }
      int down = 0;
      for (int k = 1; k <= model.staff_space / 2 && rounded_pos + k >= dst.rows;
           k++)
      {
        if (dst.at<char>(rounded_pos + k, col))
        {
          down++;
        }
        else
        {
          break;
        }
      }
      // If the CC is about the staff_height, it belongs to a staff line
      if (up + down + 1 <= model.staff_height + 1)
      {
        for (int k = 0; k <= up; k++)
        {
          dst.at<char>(rounded_pos - k, col) = 0;
        }
        for (int k = 1; k <= down; k++)
        {
          dst.at<char>(rounded_pos + k, col) = 0;
        }
      }
      // If the pixel is black, check the nearest CC
    }
    else
    {
      int up_or_down = 0;
      int start = 0;
      // Getting the nearest CC
      for (int k = 1; k <= model.staff_space / 2 &&
                      rounded_pos + k < dst.rows && rounded_pos - k >= 0;
           k++)
      {
        // up
        if (dst.at<char>(rounded_pos - k, col))
        {
          up_or_down = -1;
          start = k;
          break;
          // down
        }
        else if (dst.at<char>(rounded_pos + k, col))
        {
          up_or_down = 1;
          start = k;
          break;
        }
      }
      // If the nearest CC is small enough, it belongs to a staff line
      int cc_count = 0;
      for (int k = start; k <= model.staff_space / 2 + start &&
                          rounded_pos - k >= 0 && rounded_pos + k < dst.rows;
           k++)
      {
        if (dst.at<char>(rounded_pos + k * up_or_down, col))
        {
          cc_count++;
        }
        else
        {
          break;
        }
      }
      if (cc_count <= model.staff_height + 1)
      {
        for (int k = start; k < cc_count + start; k++)
        {
          dst.at<char>(rounded_pos + k * up_or_down, col) = 0;
        }
      }
    }
  }
}

} // namespace

StaffModel GetStaffModel(const cv::Mat &src, const int n_threads)
{
  assert(is_gray(src));
  assert(n_threads > 0);

  cv::Mat img;
  src.copyTo(img);
  if (is_black_on_white(img))
    cv::threshold(img, img, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  else
  {
    cv::threshold(img, img, 255 - BINARY_THRESH_VAL, 255, CV_THRESH_BINARY);
  }

  StaffModel model;

  // Checking whether it is straight or not
  estimate_rotation(img, model);
  if (model.straight)
  {
    std::vector<double> gradient(img.cols, 0.0);
    model.gradient = gradient;
  }

  // Getting an estimate of staff_height and staff_space
  int staff_height, staff_space;
  run_length(img, staff_height, staff_space, n_threads);

  model.staff_height = staff_height;
  model.staff_space = staff_space;

  // Removing symbols based on estimated staff_height
  cv::Mat staff_image = img;
  remove_glyphs(staff_image, staff_height, staff_space);
  model.staff_image = staff_image;
  if (model.straight)
    return model;

  estimate_gradient(model, n_threads);
  interpolate_model(model);

  return model;
}

void PrintStaffModel(cv::Mat &dst, const StaffModel &model)
{
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate(dst, dst, -rotation);
  bounding_box(dst, dst);
  if (is_gray(dst))
  {
    dst = cv::Mat(cv::Size(model.gradient.size(), model.gradient.size()),
                  CV_8UC3);
  }
  draw_model(dst, model, dst.rows / 2, cv::Scalar(255, 0, 0));
}

Staffs FitStaffModel(const StaffModel &model)
{
  cv::Mat img = model.staff_image;
  const bool straight = model.straight;

  std::vector<int> staff_lines = poll_lines(model);

  // Convolving a 1-D kernel
  // The local maximas are zones where there might be a staff
  Staffs staffs;
  const int kernel = KERNEL_SIZE * model.staff_height +
                     (KERNEL_SIZE - 1) * model.staff_space + model.staff_space;
  // higher --> harder on scarce staffs
  // lower --> staffs will start anywhere
  int min_poll_staff = 0;
  // higher --> will end anywhere
  // lower --> Harder with lyrics
  const int staff_size = (LINES_PER_STAFF - 0.5) * model.staff_height +
                         (LINES_PER_STAFF - 1) * model.staff_space;
  int min_poll_line =
      MIN_POLL_PER_LINE_RATIO * staff_lines[staff_lines.size() - 1];
  for (int i = 0; i < img.rows; i++)
  {
    int count = 0;
    for (int j = 0; j + i < img.rows && j < kernel; j++)
    {
      const int idx = i + j;
      count += staff_lines[idx];
    }
    if (count > min_poll_staff)
      min_poll_staff = count;
  }

  min_poll_staff *= POLL_PER_STAFF_RATIO;
  for (int i = 0; i < img.rows; i++)
  {
    int count = 0;
    for (int j = 0; j + i < img.rows && j < kernel; j++)
    {
      const int idx = i + j;
      count += staff_lines[idx];
    }

    if (count >= min_poll_staff)
    {
      int flag = 0;
      double start = i;
      while (i < img.rows && flag < 2 * model.staff_space)
      {
        int next_count = 0;
        for (int j = 0; j + i < img.rows && j < kernel; j++)
        {
          const int idx = j + i;
          next_count += staff_lines[idx];
        }
        if (next_count > count)
        {
          flag = 0;
          start = i;
          count = next_count;
        }
        else
        {
          flag++;
        }
        i++;
      }

      bool converged = false;
      while (!converged)
      {
        for (int k = 0; k <= model.staff_space + 2 * model.staff_height &&
                        start + k < staff_lines.size() && start - k >= 0;
             k++)
        {
          const int l = k;
          if (staff_lines[start - k] >= min_poll_line)
          {
            while (staff_lines[start - k] >= min_poll_line && start - k >= 0 &&
                   k - l <= model.staff_height)
            {
              k++;
            }
            converged = true;
            start -= (k + l - 1) / 2;
            break;
          }
          else if (staff_lines[start + k] >= min_poll_line)
          {
            while (staff_lines[start + k] >= min_poll_line &&
                   start + k < staff_lines.size() &&
                   k - l <= model.staff_height)
            {
              k++;
            }
            converged = true;
            start += (k + l - 1) / 2;
            break;
          }
        }
        min_poll_line *= 0.9;
      }
      const int finish = start + staff_size;
      i = finish + model.staff_space;
      staffs.push_back(std::pair<int, int>(start, finish));
      min_poll_line =
          MIN_POLL_PER_LINE_RATIO * staff_lines[staff_lines.size() - 1];
    }
  }

  if (remove)
  {
  }

  return staffs;
}

void PrintStaffs(cv::Mat &dst, const Staffs &staffs,
                 const StaffModel &model)
{
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate(dst, dst, rotation);
  if (is_gray(dst))
  {
    cv::cvtColor(dst, dst, CV_GRAY2BGR);
  }
  for (auto s : staffs)
  {
    const double staff_interval = s.second - s.first;
    for (int i = 1; i < LINES_PER_STAFF - 1; i++)
    {
      const int line_pos = round(staff_interval / (LINES_PER_STAFF - 1) * i) +
                           s.first + model.start_row;
      draw_model(dst, model, line_pos, cv::Scalar(255, 0, 0));
    }
    int line_pos = s.first + model.start_row;
    draw_model(dst, model, line_pos, cv::Scalar(0, 255, 0));
    line_pos =
        round(staff_interval / (LINES_PER_STAFF - 1) * (LINES_PER_STAFF - 1)) +
        s.first + model.start_row;
    draw_model(dst, model, line_pos, cv::Scalar(0, 0, 255));
  }
}

void RemoveStaffs(cv::Mat &dst, const Staffs &staffs,
                  const StaffModel &model)
{
  assert(is_gray(dst));
  const bool blackOnWhite = is_black_on_white(dst);
  if (blackOnWhite)
    cv::threshold(dst, dst, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  else
  {
    cv::threshold(dst, dst, 255 - BINARY_THRESH_VAL, 255, CV_THRESH_BINARY);
  }
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate(dst, dst, rotation);
  for (auto it = staffs.begin(); it != staffs.end(); it++)
  {
    const double staff_interval =
        (it->second - it->first) / (LINES_PER_STAFF - 1);
    for (int i = 0; i < LINES_PER_STAFF; i++)
    {
      const int line_pos =
          round(staff_interval * i) + it->first + model.start_row;
      remove_line(dst, line_pos, model);
      for (int j = 1; j <= 1; j++)
      {
        remove_line(dst, line_pos + j, model);
        remove_line(dst, line_pos - j, model);
      }
    }
  }
  if (blackOnWhite)
    cv::threshold(dst, dst, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  raw_rotate(dst, dst, -rotation);
}

void Realign(cv::Mat &dst, const StaffModel &model)
{
  assert(is_gray(dst));
  assert(dst.cols >= model.gradient.size());

  if (model.straight)
  {
    return;
  }

  const bool blackOnWhite = is_black_on_white(dst);
  if (blackOnWhite)
    cv::threshold(dst, dst, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  else
  {
    cv::threshold(dst, dst, 255 - BINARY_THRESH_VAL, 255, CV_THRESH_BINARY);
  }
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate(dst, dst, rotation);

  cv::Mat img = dst.clone();

  double start = 0;
  for (int x = 0; x < model.gradient.size(); x++, start += model.gradient[x])
  {
    const int rounded_start = round(start);
    const int col = x + model.start_col;
    for (int y = 0; y < img.rows; y++)
    {
      const int offset_y = y + rounded_start;
      if (offset_y >= img.rows || offset_y < 0)
      {
        img.at<char>(y, col) = 0;
      }
      else
      {
        img.at<char>(y, col) = dst.at<char>(offset_y, col);
      }
    }
  }
  dst = img;
  if (blackOnWhite)
    cv::threshold(dst, dst, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  raw_rotate(dst, dst, -rotation);
}

void SaveToDisk(const std::string &fn, const Staffs &staffs,
                const StaffModel &model)
{
  using namespace tinyxml2;
  XMLDocument doc;

  // <Autoscore>
  XMLNode *Root = doc.NewElement("stav");
  doc.InsertFirstChild(Root);

  // <filename>
  XMLElement *Filename = doc.NewElement("filename");
  Filename->SetText((strip_fn(fn)).c_str());
  Root->InsertFirstChild(Filename);

  // <Model...>
  XMLNode *Model = doc.NewElement("model");
  Root->InsertAfterChild(Filename, Model);

  // <staff_height, staff_space, etc.>
  XMLElement *StaffHeight = doc.NewElement("staff_height");
  StaffHeight->SetText(std::to_string(model.staff_height).c_str());
  Model->InsertFirstChild(StaffHeight);

  XMLElement *StaffSpace = doc.NewElement("staff_space");
  StaffSpace->SetText(std::to_string(model.staff_space).c_str());
  Model->InsertAfterChild(StaffHeight, StaffSpace);

  XMLElement *StartCol = doc.NewElement("start_column");
  StartCol->SetText(std::to_string(model.start_col).c_str());
  Model->InsertAfterChild(StaffSpace, StartCol);

  XMLElement *StartRow = doc.NewElement("start_row");
  StartRow->SetText(std::to_string(model.start_row).c_str());
  Model->InsertAfterChild(StartCol, StartRow);

  XMLElement *Rotation = doc.NewElement("rotation");
  Rotation->SetText(std::to_string(model.rot).c_str());
  Model->InsertAfterChild(StartRow, Rotation);

  XMLElement *Gradient = doc.NewElement("gradient");
  std::string grad;
  for (double g : model.gradient)
  {
    grad += std::to_string(g) + ' ';
  }
  Gradient->SetText(grad.c_str());
  Model->InsertAfterChild(Rotation, Gradient);

  // <staffs>
  XMLNode *AllStaffs = doc.NewElement("staffs");
  Root->InsertAfterChild(Model, AllStaffs);

  for (auto staff : staffs)
  {
    XMLElement *s = doc.NewElement("staff");
    s->SetText(std::to_string(staff.first).c_str());
    AllStaffs->InsertEndChild(s);
  }

  assert(doc.SaveFile((strip_ext(fn) + ".xml").c_str()) >= 0);
}