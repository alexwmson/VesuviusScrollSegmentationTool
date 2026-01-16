#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::string tif_dir = argv[1];
    std::string png_dir = argv[2];

    fs::create_directories(png_dir);

    for (auto & entry : fs::directory_iterator(tif_dir)) {
        if (entry.path().extension() == ".tif") {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
            std::string png_path = png_dir + "/" + entry.path().stem().string() + ".png";
            cv::imwrite(png_path, img);
        }
    }
    return 0;
}