#include <staff.hh>
#include <util.hh>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <string>
#include <thread>
#include <list>
#include <unordered_set>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

typedef struct request
{
  std::list<std::string> filenames;
  bool out_xml, out_image, rectify, verbose, remove_staff;
  std::string output_dir;
} Request;

void process_request(const Request &r)
{
  for (auto fn : r.filenames)
  {
    cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
    to_binary(img, img);
    auto model = GetStaffModel(img);
    auto staffs = FitStaffModel(model);
    if (r.rectify)
      Realign(img, model);
    if (r.out_image)
    {
      PrintStaffs(img, staffs, model);
      fs::path p(r.output_dir);
      p /= strip_fn(fn);
      cv::imwrite(p.generic_string(), img);
    }
    if (r.out_xml)
    {
      fs::path p(r.output_dir);
      p /= strip_fn(fn);
      SaveToDisk(p.generic_string(), staffs, model);
    }
    if(r.remove_staff)
    {
      cv::Mat cleared;
      img.copyTo(cleared);
      cv::cvtColor(cleared, cleared, CV_BGR2GRAY); // 
      RemoveStaffs(cleared, staffs, model);
      fs::path p(r.output_dir);
      p /= std::string("cleared_") + strip_fn(fn);
      cv::imwrite(p.generic_string(), cleared);      
    }
    if (r.verbose)
      std::cout << fn << std::endl;
  }
}

typedef struct global_request
{
  bool xml, image, rectify, batch, verbose, remove_staff;
  std::string input_dir, input_file, output_dir;
  int n_files, n_threads;
} GlobalRequest;

void delegate_tasks(GlobalRequest r)
{
  std::list<std::string> files;
  if (!r.batch)
  {
    Request req = {files, r.xml, r.image, r.rectify, r.verbose, r.remove_staff, r.output_dir};
    files.push_back(r.input_file);
    req.filenames = files;
    process_request(req);
  }
  else 
  {
    std::unordered_set<std::string> existing_files;
    fs::path in(r.input_dir);
    fs::path out(r.output_dir);
    for (auto &p : fs::directory_iterator(out))
    {
      std::string fn = p.path().filename().string();
      if (fs::is_regular_file(p) && is_image(fn))
        existing_files.emplace(fn);
      
    }
    for (auto &p : fs::directory_iterator(in))
    {
      std::string fn = p.path().filename().string();
      if (!existing_files.count(fn) && is_image(fn))
      {
        fs::path d(r.input_dir);
        d /= fn;
        files.push_back(d.generic_string());
      }
    }

    if (files.empty()) return;
    auto list_iterator = files.begin();
    if (r.n_files == -1)
      r.n_files = files.size();
    const int real_n_files = r.n_files > files.size() ? files.size() : r.n_files;
    const int files_per_thread = real_n_files / r.n_threads;
    std::vector<std::thread> threads(r.n_threads);
    std::cout << files.size() << std::endl;
    for (int i = 0; i < r.n_threads; ++i)
    {
      const auto start = list_iterator;
      const int n_files = i == r.n_threads - 1 ? (files_per_thread + files.size() % r.n_threads) : files_per_thread;
      for (int j = 0; j < n_files; ++j)
      {
        list_iterator++;
      }
      std::list<std::string> thread_list(start, list_iterator);
      Request req = {thread_list, r.xml, r.image, r.rectify, r.verbose, r.remove_staff, r.output_dir};
      threads[i] = std::thread(process_request, req);
    }
    for (auto &t : threads)
      t.join();
  }
}

int main(int argc, char **argv)
{
  po::options_description description("Allowed options");
  description.add_options()("help,h", "Produces help message");
  description.add_options()("batch,b", po::value<std::string>(), "Processes all files in specified directory");
  description.add_options()("number,n", po::value<int>()->default_value(-1), "Number of files to process");
  description.add_options()("xml,x", "Outputs an XML description of the staff model");
  description.add_options()("image,j", "Outputs an annotated image");
  description.add_options()("out_directory,o", po::value<std::string>()->default_value("."), "Output directory");
  description.add_options()("rectify,r", "Rectifies the model to be straight");
  description.add_options()("input,i", po::value<std::string>(), "Input music score file");
  description.add_options()("n_threads,t", po::value<int>()->default_value(1), "Number of threads");
  description.add_options()("verbose,v", "Talk to me baby");
  description.add_options()("remove_staff,c", "Clears the input image of all detected staffs");

  po::variables_map variable_map;
  po::store(po::parse_command_line(argc, argv, description), variable_map);
  po::notify(variable_map);

  bool xml, image, batch, rectify, verbose, remove_staff;
  std::string input_dir, input_file, output_dir;
  int n_files, n_threads;

  n_files = variable_map["number"].as<int>();
  output_dir = variable_map["out_directory"].as<std::string>();
  n_threads = variable_map["n_threads"].as<int>();

  if (variable_map.count("help"))
  {
    std::cout << description << "\n";
    return 1;
  }
  if (variable_map.count("xml"))
  {
    xml = true;
  }
  if (variable_map.count("image"))
  {
    image = true;
  }
  if (!image && !xml)
  {
    std::cout << "There is nothing to output. Chose an image or an xml in the options\n";
    return -1;
  }
  if (variable_map.count("verbose"))
  {
    verbose = true;
  }
  if (variable_map.count("remove_staff"))
  {
    remove_staff = true;
  }
  if (variable_map.count("rectify"))
  {
    rectify = true;
  }
  if (variable_map.count("batch"))
  {
    if (variable_map.count("input"))
    {
      std::cout << "Batch processing and input files are mutually exclusive\n";
      return -1;
    }
    batch = true;
    input_dir = variable_map["batch"].as<std::string>();
  }
  else if (variable_map.count("input"))
  {
    if (n_files != -1)
    {
      std::cout << "Will only process one file because the batch option was not given.\n";
    }
    if (n_threads > 1)
    {
      std::cout << "Will only use one thread for the file\n";
    }
    input_file = variable_map["input"].as<std::string>();
  }

  GlobalRequest req = {xml, image, rectify, batch, verbose, remove_staff, input_dir, input_file, output_dir, n_files, n_threads};
  delegate_tasks(req);

  return 0;
}