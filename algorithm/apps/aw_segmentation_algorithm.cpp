#include <cstdint>

#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <omp.h>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <unordered_map>
#include <climits>


//#define INDEX(x, y) (x) + (y) * width

namespace {
    // From vc_grow_seg_from_seed.cpp
    // I did not write this, it's handy though
    std::string time_str()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        auto timer = system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&timer);
        
        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y%m%d%H%M%S");
        oss << std::setfill('0') << std::setw(3) << ms.count();

        return oss.str();
    }

    
    uint8_t get_val(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp, uint16_t x, uint16_t y, uint16_t z) {
        double v = 0;
        interp.Evaluate(static_cast<double>(z), static_cast<double>(y), static_cast<double>(x), &v);
        return static_cast<uint8_t>(v);
    }

    const std::array<cv::Vec3f, 6> directions6 = {{
    cv::Vec3f(-1.0f,  0.0f,  0.0f), // left
    cv::Vec3f( 1.0f,  0.0f,  0.0f), // right
    cv::Vec3f( 0.0f, -1.0f,  0.0f), // down
    cv::Vec3f( 0.0f,  1.0f,  0.0f), // up
    cv::Vec3f( 0.0f,  0.0f, -1.0f), // back
    cv::Vec3f( 0.0f,  0.0f,  1.0f)  // front
    }};

    const std::array<std::array<int, 3>, 26> directions26 = {{
    {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
    {-1,  0, -1}, {-1,  0, 0}, {-1,  0, 1},
    {-1,  1, -1}, {-1,  1, 0}, {-1,  1, 1},

    {0, -1, -1},  {0, -1, 0},  {0, -1, 1},
    {0,  0, -1},                {0,  0, 1},
    {0,  1, -1},  {0,  1, 0},  {0,  1, 1},

    {1, -1, -1},  {1, -1, 0},  {1, -1, 1},
    {1,  0, -1},  {1,  0, 0},  {1,  0, 1},
    {1,  1, -1},  {1,  1, 0},  {1,  1, 1}
    }};

    const std::array<std::array<int, 2>, 8> directions8 = {{
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1}
    }};

    const std::array<std::array<int, 3>, 18> directions18 = {{
    // Faces (6)
    {-1,  0,  0}, {1, 0, 0},
    { 0, -1,  0}, {0, 1, 0},
    { 0,  0, -1}, {0, 0, 1},

    // Edges (12)
    {-1, -1,  0}, {-1, 1,  0}, {1, -1,  0}, {1, 1,  0},
    {-1,  0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
    { 0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1}
    }};
    
    uint8_t GlobalThreshold; //Threshold used to determine if voxel is scroll or air
    uint16_t volumeSizeX;
    uint16_t volumeSizeY;
    uint16_t volumeSizeZ;
    int stepSize;

    float allowedDifference; //90 degrees
}

struct point{
     //The value of the voxel (0-255)
    uint8_t value = 0;

    //positions in 3d volume
    uint16_t x = 0, y = 0, z = 0;
    /*
    Bitmask for all 6 directions
    Basically saying: 1 = yep that neighboring voxel there is a thing, 0 = nope thats empty space
    Stored like a stack
    */
    uint8_t neighbors = 0b1000000;
    
    cv::Vec3f normal;
    cv::Vec3f normalizedNormal;
    cv::Vec3f plane;

    int i = 0, j = 0;

    bool ignore = false;
    uint8_t color = 255;

    void computeNeighbors(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp);
    void computeNormal(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp);
    bool isSurface(CachedChunked3dInterpolator<uint8_t, passTroughComputor>& interp);
    bool isBridgePoint(CachedChunked3dInterpolator<uint8_t, passTroughComputor>& interp);
};

point findStart(CachedChunked3dInterpolator<uint8_t, passTroughComputor>& interp, uint16_t startX, uint16_t startY, uint16_t startZ);
void grow(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp, std::unordered_map<uint64_t, std::shared_ptr<point>> &points, point startingPoint, uint8_t rNormalTimerMax, uint8_t patienceMax, int maxLayers, uint64_t maxSize);
bool isWithinVolume(uint16_t x, uint16_t y, uint16_t z);

inline bool safeAdd(uint16_t base, int delta, uint16_t &result);
inline uint64_t combineCoords(uint16_t x, uint16_t y, uint16_t z);
inline void updateTangents(cv::Vec3f &oldNormal, cv::Vec3f &newNormal, cv::Vec3f &t1, cv::Vec3f &t2);

cv::Mat_<cv::Vec3f> downsampleGrid(const cv::Mat_<cv::Vec3f>& grid,int factor);
inline void fillHoles(cv::Mat_<cv::Vec3f>& grid, int invalidCellCount);

void point::computeNeighbors(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp) {
    neighbors = 0;
    for (size_t i = 0; i < 6; i++) {
        uint16_t nx = x + directions6[i][0];
        uint16_t ny = y + directions6[i][1];
        uint16_t nz = z + directions6[i][2];
        if (isWithinVolume(nx, ny, nz)) {
            double value = get_val(interp, nx, ny, nz);
            neighbors |= (value >= GlobalThreshold) << i;  // Bit i -> directions6[i]
        }
    }
}

void point::computeNormal(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp) {
    normal = cv::Vec3f(
        static_cast<float>((get_val(interp, x - 1, y, z) - get_val(interp, x + 1, y, z)) / 2),
        static_cast<float>((get_val(interp, x, y - 1, z) - get_val(interp, x, y + 1, z)) / 2),
        static_cast<float>((get_val(interp, x, y, z - 1) - get_val(interp, x, y, z + 1)) / 2)
    );
    cv::normalize(normal, normalizedNormal);
}

bool point::isSurface(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp) {
    if (value < GlobalThreshold)
        return false;
    if (neighbors == 0b1000000)
        computeNeighbors(interp);
    return neighbors != 0b111111;
}
int main(int argc, char *argv[]){
    
    std::filesystem::path vol_path, tgt_dir, params_path, resume_path, correct_path;
    cv::Vec3d origin;
    nlohmann::json params;

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("volume,v", boost::program_options::value<std::string>()->required(), "OME-Zarr volume path, the raw data not the surface prediction")
        ("target-dir,t", boost::program_options::value<std::string>()->required(), "Target directory for output")
        ("params,p", boost::program_options::value<std::string>(), "JSON parameters file")
        ("seed,s", boost::program_options::value<std::vector<float>>()->multitoken(), "Seed coordinates (x y z)");
 
    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }

        boost::program_options::notify(vm);
    } catch (const boost::program_options::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("params")) {
        std::ifstream params_f(vm["params"].as<std::string>());
        if (!params_f.is_open()) {
            std::cerr << "ERROR: Could not open parameters file" << std::endl;
            return EXIT_FAILURE;
        }
        try {
            params = nlohmann::json::parse(params_f);
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "ERROR: Failed to parse JSON: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        params = nlohmann::json::object();
    }

    vol_path = vm["volume"].as<std::string>();
    tgt_dir = vm["target-dir"].as<std::string>();

    // Params stuff
    GlobalThreshold = params.value("global_threshold", 40);  //Threshold used to determine if voxel is scroll or air
    allowedDifference = params.value("allowed_difference", 0); // 0 -> 90 degrees, uses cross product
    uint8_t patienceMax = params.value("max_patience", 5); // The max the patience counter can reach. I.e the most comfortable the bfs can be, dont depend on this it will probably be removed / replaced by min/max size 
    int maxLayers = params.value("max_layers", 2000); // Maximum number of bfs loops. If the code produced a perfect sphere, maxLayers is the radius
    uint64_t maxSize = params.value("max_size", (uint64_t)250000); // Maximum points allowed, defaults to 250k
    int minSize = params.value("min_size", 50000); // Only used during random seed, used to determine min island size
    float voxelSize = params.value("scale", 7.81); // The scrolls scale, defaults to 7.81 microns
    int stepSize = params.value("steps", 10); // Number of voxels to go over by
    std::string time = time_str();
    std::string uuid = params.value("uuid", "segment_" + time); // uuid for folder name
    cv::Vec2f scale(voxelSize, voxelSize);

    std::cout << "GlobalThreshold: " << static_cast<int>(GlobalThreshold )<< std::endl;
    std::cout << "allowedDifference: " << allowedDifference << std::endl;
    std::cout << "patienceMax: " << static_cast<int>(patienceMax) << std::endl;
    std::cout << "maxLayers: " << maxLayers << std::endl;
    std::cout << "maxSize: " << maxSize << std::endl;
    if (params.contains("min_size"))
        std::cout << "minSize: " << minSize << std::endl;

    // Just chunk management stuff, important part is interpolator is the object which gets values
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);

    nlohmann::json zarray = nlohmann::json::parse(std::ifstream(vol_path/"0/.zarray"));
    z5::filesystem::handle::Dataset ds_handle(group, "0", zarray.value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    ChunkCache<uint8_t> chunk_cache(static_cast<size_t>(params.value("cache_size", 1e9)));

    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> tensor(pass, ds.get(), &chunk_cache);
    CachedChunked3dInterpolator<uint8_t,passTroughComputor> interpolator(tensor);

    std::vector<uint64_t> shape = zarray.at("shape").get<std::vector<uint64_t>>();

    if (shape[0] > UINT16_MAX || shape[1] > UINT16_MAX || shape[2] > UINT16_MAX) {
        std::cerr << "ERROR: This code only works for volumes where the dimensions are less than 2^16 (must be 0 - 65535)" << std::endl;
        return EXIT_FAILURE;
    }

    volumeSizeZ = static_cast<uint16_t>(shape[0]);
    volumeSizeY = static_cast<uint16_t>(shape[1]);
    volumeSizeX = static_cast<uint16_t>(shape[2]);

    if (volumeSizeX == 0 || volumeSizeY == 0 || volumeSizeZ == 0) {
        std::cerr << "ERROR: Atleast one volume dimension is 0" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Volume size, Z: " << volumeSizeZ << ", Y: " << volumeSizeY << ", X: " << volumeSizeX << std::endl;
    point startingPoint{};
    if (vm.count("seed")){
        auto seedVec = vm["seed"].as<std::vector<float>>();
        uint16_t x = seedVec[0], y = seedVec[1], z = seedVec[2];
        std::cout << "Starting position, x: " << x << ", y: " << y << ", z: " << z << std::endl;
        startingPoint = findStart(interpolator, x, y, z);
        
        // Found that a lot of times the code finds little 1x1x1 islands or something similar
        // Nobody wants these, so the code tries to grow out at most maxLayers (defaults to 50) bfs layers to make sure it's not on one of these crappy islands
        std::unordered_map<uint64_t, std::shared_ptr<point>> tempPoints;
        grow(interpolator, tempPoints, startingPoint, 5, patienceMax, maxLayers, static_cast<uint64_t>(minSize));  //rNUT set to 5 since it's not calculated yet

        if (startingPoint.value == 0 || tempPoints.size() < minSize){
            int maxRadius = 100, growth = 10;
            const int sx = x, sy = y, sz = z;
            std::atomic<int> bestTry(1000);
            std::mutex sp_mutex;

            #pragma omp parallel for
            for (int tries = 0; tries < 1000; tries++) {
                if (bestTry.load() < tries) continue; // already found lowest try
                std::mt19937 rng(static_cast<uint32_t>(x * 31 ^ y * 37 ^ z * 41 ^ tries));

                int radius = std::min(maxRadius, 1 + tries / growth);
                int dx, dy, dz;
                do {
                    std::uniform_int_distribution<int> dist(-radius, radius);
                    dx = dist(rng);
                    dy = dist(rng);
                    dz = dist(rng);
                } while (dx*dx + dy*dy + dz*dz > radius*radius);

                int nx = sx + dx;
                int ny = sy + dy;
                int nz = sz + dz;

                if (nx < 0 || ny < 0 || nz < 0 || nx >= volumeSizeX || ny >= volumeSizeY || nz >= volumeSizeZ)
                    continue;

                std::cout << "Starting position, x: " << nx << ", y: " << ny << ", z: " << nz << std::endl;

                point candidate = findStart(interpolator, nx, ny, nz);
                if (candidate.value == 0) continue;

                // Found that a lot of times the code finds little 1x1x1 islands or something similar
                // Nobody wants these, so the code tries to grow out at most maxLayers (defaults to 50) bfs layers to make sure it's not on one of these crappy islands
                std::unordered_map<uint64_t, std::shared_ptr<point>> tempPoints;
                grow(interpolator, tempPoints, candidate, 5, patienceMax, maxLayers, minSize);  //rNUT set to 5 since it's not calculated yet

                if (tempPoints.size() >= minSize) {
                    int curBest = bestTry.load();
                    while (tries < curBest) {
                        if (bestTry.compare_exchange_weak(curBest, tries)) {
                            std::lock_guard<std::mutex> lock(sp_mutex);
                            startingPoint = candidate;
                            break;
                        }
                    }
                }
            }
        }
    }
    else{
        int maxTries = 1000;
        std::atomic<int> bestTry(maxTries);
        std::mutex sp_mutex; // protects startingPoint

        #pragma omp parallel for
        for (int tries = 0; tries < maxTries; tries++){
            if (bestTry.load() < tries) continue;

            std::mt19937 rng(std::random_device{}() + omp_get_thread_num());
            std::uniform_int_distribution<uint16_t> posX(0, volumeSizeX - 1);
            std::uniform_int_distribution<uint16_t> posY(0, volumeSizeY - 1);
            std::uniform_int_distribution<uint16_t> posZ(0, volumeSizeZ - 1);
            uint16_t x = posX(rng), y = posY(rng), z = posZ(rng);

            std::cout << "Starting position, x: " << x << ", y: " << y << ", z: " << z << std::endl;
            point candidate = findStart(interpolator, x, y, z);
            if (candidate.value == 0) continue;

            // Found that a lot of times the code finds little 1x1x1 islands or something similar
            // Nobody wants these, so the code tries to grow out at most maxLayers (defaults to 50) bfs layers to make sure it's not on one of these crappy islands
            std::unordered_map<uint64_t, std::shared_ptr<point>> tempPoints;
            grow(interpolator, tempPoints, candidate, 5, patienceMax, maxLayers, static_cast<uint64_t>(minSize)); //rNUT set to 5 since it's not calculated yet
            
            if (tempPoints.size() >= minSize) {
                int curBest = bestTry.load();
                while (tries < curBest) {
                    if (bestTry.compare_exchange_weak(curBest, tries)) {
                        std::lock_guard<std::mutex> lock(sp_mutex);
                        startingPoint = candidate;
                        break;
                    }
                }
            }
        }
    }
    if (startingPoint.value == 0) {
        std::cerr << "WARNING: Aw man, could not find a surface point.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Starting point found!" << std::endl;
    std::cout << "Point value: " << static_cast<int>(startingPoint.value) << std::endl;
    std::cout << "Point x coord: " << startingPoint.x << std::endl;
    std::cout << "Point y coord: " << startingPoint.y << std::endl;
    std::cout << "Point z coord: " << startingPoint.z << std::endl;
    std::cout << "Starting neighbors: " << std::bitset<8>(startingPoint.neighbors) << std::endl;

    float dx = (startingPoint.x - volumeSizeX / 2), dy = (startingPoint.y - volumeSizeY / 2), dz = (startingPoint.z - volumeSizeZ / 2);
    float distanceFromCenter = sqrt(dx*dx + dy*dy + dz*dz);
    uint8_t referenceNormalUpdateTimer = std::max(5.0f, sqrt(distanceFromCenter)); // 900 pixels away should recalculate every roughly 30 voxels or so, and 100 pixels away should recalculate every 10. based on what Ive seen

    std::unordered_map<uint64_t, std::shared_ptr<point>> points;

    std::cout << "Distance from center: " << distanceFromCenter << std::endl;
    std::cout << "referenceNormalUpdateTimer: " << static_cast<int>(referenceNormalUpdateTimer) << std::endl;
    std::cout << "Starting Grow function" << std::endl;

    grow(interpolator, points, startingPoint, referenceNormalUpdateTimer, patienceMax, maxLayers, maxSize);

    std::cout << "Grow function finished" << std::endl;
    std::cout << "Total number of points: " << points.size() << std::endl;
    std::cout << "Time to turn this thing into a 2d array" << std::endl;

    int min_i = INT_MAX;
    int max_i = INT_MIN;
    int min_j = INT_MAX;
    int max_j = INT_MIN;

    for (auto& [_, p] : points) {
        min_i = std::min(min_i, p->i);
        max_i = std::max(max_i, p->i);
        min_j = std::min(min_j, p->j);
        max_j = std::max(max_j, p->j);
    }

    int width  = max_i - min_i + 1;
    int height = max_j - min_j + 1;

    cv::Mat_<cv::Vec3f> grid(height, width, cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> colorGrid(height, width, static_cast<uint8_t>(0));

    for (auto& [k, p] : points) {
        if (p->ignore)
            continue;
        int u = p->i - min_i;
        int v = p->j - min_j;
        grid(v, u) = cv::Vec3f(p->x, p->y, p->z);
        colorGrid(v, u) = p->color;
    }

    // Save as tifxyz
    auto downsampled = downsampleGrid(grid, stepSize);

    int valid = 0;
    for (int r = 0; r < downsampled.rows; ++r) {
        for (int c = 0; c < downsampled.cols; ++c) {
            if (downsampled(r,c) != cv::Vec3f(-1,-1,-1))
                ++valid;
        }
    }

    std::cout << "Valid cells: " << valid
            << " / " << (downsampled.rows * downsampled.cols)
            << std::endl;

    //fillHoles(downsampled, downsampled.rows * downsampled.cols - valid);

    valid = 0;
    for (int r = 0; r < downsampled.rows; ++r) {
        for (int c = 0; c < downsampled.cols; ++c) {
            if (downsampled(r,c) != cv::Vec3f(-1,-1,-1))
                ++valid;
        }
    }

    std::cout << "Valid cells again: " << valid
            << " / " << (downsampled.rows * downsampled.cols)
            << std::endl;

    QuadSurface surf(downsampled, scale / static_cast<float>(stepSize));
    std::filesystem::path segment_dir = tgt_dir / uuid;

    double area_vx2 = vc::surface::computeSurfaceAreaVox2(downsampled);
    double area_cm2 = area_vx2 * voxelSize * voxelSize / 1e8;
    std::cout << "Area (cm^2): " << area_cm2 << std::endl;

    // Print some statistics
    int filledCells = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (grid(row, col) != cv::Vec3f(-1, -1, -1)) {
                filledCells++;
            }
        }
    }

    std::cout << "Filled cells: " << filledCells << " / " << (width * height) << " (" << (100.0 * filledCells / (width * height)) << "%)" << std::endl;
    std::cout << "Average overlap (I.e avg of x points being assigned to n cells in the grid): " << static_cast<float>(points.size()) / static_cast<float>(filledCells) << std::endl;

    nlohmann::json segment_meta;
    segment_meta["source"] = "aw_segmentation_algorithm";
    segment_meta["area_cm2"] = area_cm2;
    segment_meta["area_vx2"] = area_vx2;
    segment_meta["params_used"] = params;
    segment_meta["volume"] = std::filesystem::path(vol_path).filename().string();
    segment_meta["average_overlap"] = static_cast<float>(points.size()) / static_cast<float>(filledCells);
    segment_meta["starting_point"] = {startingPoint.x, startingPoint.y, startingPoint.z};
    segment_meta["pixels_on_grayscale"] = filledCells;
    segment_meta["voxels_explored"] = points.size();
    segment_meta["created_on"] = time;

    surf.meta = std::make_unique<nlohmann::json>(std::move(segment_meta));

    try {
        surf.save(segment_dir, uuid);
        std::cout << "Saved tifxyz to " << tgt_dir  << "/" << uuid << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving tifxyz: " << e.what() << std::endl;
    }

    // Save the grid as an image to visualize
    std::cout << "Grid dimensions: " << width << " x " << height << std::endl;
    cv::Mat visualization(height, width, CV_8UC3, cv::Scalar(50, 50, 50));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++){
        if (grid.at<cv::Vec3f>(i, j) == cv::Vec3f(-1,-1,-1))
            continue;
        uint8_t color = colorGrid(i, j);
        visualization.at<cv::Vec3b>(i, j) = cv::Vec3b(color, color, color);
        }
    }

    std::filesystem::path filename = segment_dir / ("grayscale.png");
    cv::imwrite(filename.string(), visualization);
    std::cout << "Saved grid visualization to " << filename.string() << std::endl;

    /*
    for (int y = 0; y < downsampled.rows; y++) {
        for (int x = 0; x < downsampled.cols; x++) {

            const cv::Vec3f& v = downsampled(y, x);

            // Print each cell as (x,y,z)
            std::cout << "("
                      << std::setw(6) << std::fixed << std::setprecision(2) << v[0] << ", "
                      << std::setw(6) << v[1] << ", "
                      << std::setw(6) << v[2] << ") ";
        }
        std::cout << "\n"; // new row
    }*/

    
    return EXIT_SUCCESS;
}

inline bool safeAdd(uint16_t base, int delta, uint16_t &result) {
    int val = static_cast<int>(base) + delta;
    if (val < 0 || val > UINT16_MAX)
        return false;
    result = static_cast<uint16_t>(val);
    return true;
}

inline uint64_t combineCoords(uint16_t x, uint16_t y, uint16_t z) {
    // Pack x, y, z into 64 bits: [x:16][y:16][z:16]
    return (static_cast<uint64_t>(x) << 32) | (static_cast<uint64_t>(y) << 16) | static_cast<uint64_t>(z);
}

point findStart(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp, uint16_t startX, uint16_t startY, uint16_t startZ){
    if (!isWithinVolume(startX, startY, startZ))
        return point{};

    std::queue<point> q;
    bool seen[100][100][100] = {false};
    uint16_t newX, newY, newZ;
    int indexX, indexY, indexZ;
    int offsetX = startX > 50 ? startX - 50 : 0, offsetY = startY > 50 ? startY - 50 : 0, offsetZ = startZ > 50 ? startZ - 50 : 0;

    q.push(point{0, startX, startY, startZ});
    seen[startX - offsetX][startY - offsetY][startZ - offsetZ] = true;

    while (!q.empty()){
        point p = q.front();
        q.pop();

        p.value = get_val(interp, p.x, p.y, p.z);
        if (p.isSurface(interp))
            return p;

        for (const auto &neighbor : directions26){
            if (!safeAdd(p.x, neighbor[0], newX)) continue;
            if (!safeAdd(p.y, neighbor[1], newY)) continue;
            if (!safeAdd(p.z, neighbor[2], newZ)) continue;

            indexX = newX - offsetX; indexY = newY - offsetY; indexZ = newZ - offsetZ;
            if (isWithinVolume(newX, newY, newZ) && (indexX > -1 && indexX < 100 && indexY > -1 && indexY < 100 && indexZ > -1 && indexZ < 100) && !seen[indexX][indexY][indexZ]){
                seen[indexX][indexY][indexZ] = true;
                q.push(point{0, newX, newY, newZ});
            }
        }
    }
    return point{};
}

bool isWithinVolume(uint16_t x, uint16_t y, uint16_t z){
    return x < volumeSizeX && y < volumeSizeY && z < volumeSizeZ;
}

void grow(CachedChunked3dInterpolator<uint8_t, passTroughComputor> &interp, std::unordered_map<uint64_t, std::shared_ptr<point>> &points, point startingPoint, uint8_t rNormalTimerMax, uint8_t patienceMax, int maxLayers, uint64_t maxSize){
    std::unordered_set<int64_t> ij_points;
    cv::Vec3f globalOrigin(startingPoint.x, startingPoint.y, startingPoint.z);
    std::queue<std::tuple<std::shared_ptr<point>, cv::Vec3f, cv::Vec3f, cv::Vec3f, uint8_t, uint8_t>> q;
    std::unordered_set<uint64_t> seen;
    int loops = 1;

    startingPoint.computeNormal(interp);
    if (cv::norm(startingPoint.normal) == 0) return; //If this happens you should buy a lottery ticket

    cv::Vec3f t1, t2;
    cv::Vec3f refNormal = startingPoint.normalizedNormal;

    if (std::abs(refNormal[0]) > 0.1f) {
        t1 = cv::Vec3f(0, 1, 0);
    } else {
        t1 = cv::Vec3f(1, 0, 0);
    }
    t1 = t1 - refNormal * refNormal.dot(t1);
    t1 = cv::normalize(t1);
    t2 = cv::normalize(refNormal.cross(t1));
    
    q.push(std::make_tuple(std::make_shared<point>(startingPoint), refNormal, t1, t2, patienceMax, rNormalTimerMax));
    seen.insert(combineCoords(startingPoint.x, startingPoint.y, startingPoint.z));

    while (!q.empty() && loops++ <= maxLayers){
        int size = q.size();
        for (int batch = 0; batch < size; batch++){
            // p is the point obviously
            // referenceNormal is used to make sure the surface isn't growing onto other pages by ensuring that the normal doesn't change too much, it's also normalized
            // updateNormalTimer is how long until referenceNormal should be recalculated
            // patience is how many more voxels (with normals different enough from referenceNormal) are allowed before stopping
            auto [pPointer, referenceNormal, t1, t2, patience, updateNormalTimer] = q.front();
            q.pop();

            point &p = *pPointer;

            float refMag = cv::norm(referenceNormal);
            float pMag = cv::norm(p.normalizedNormal);

            float normalDifference = referenceNormal.dot(p.normalizedNormal) / (refMag * pMag);
            if (normalDifference < allowedDifference) //0 degrees == 1, 180 degrees == -1
                patience--;
            else
                if (patience < patienceMax)
                    patience++;
            
            p.color = static_cast<uint8_t>(std::round((std::clamp(normalDifference, -1.0f, 1.0f) + 1.0f) / 2.0f * 255.0f));

            if (patience > 0){
                if (patience == patienceMax && updateNormalTimer == 0){ //Only update referenceNormal if it's time to and we're not in a weird part of the scroll (A rough part for example)
                    updateTangents(referenceNormal, p.normalizedNormal, t1, t2);
                    referenceNormal = p.normalizedNormal;
                    updateNormalTimer = rNormalTimerMax;
                }

                points[combineCoords(p.x, p.y, p.z)] = pPointer;

                if (points.size() >= maxSize) // Return early if we already hit maxSize
                    return;

                for (const auto &neighbor : directions18){
                    uint16_t newX = 0, newY = 0, newZ = 0;
                    if (!safeAdd(p.x, neighbor[0], newX)) continue;
                    if (!safeAdd(p.y, neighbor[1], newY)) continue;
                    if (!safeAdd(p.z, neighbor[2], newZ)) continue;

                    if (isWithinVolume(newX, newY, newZ) && seen.find(combineCoords(newX, newY, newZ)) == seen.end()){
                        // The new voxel must follow the 3 commandments
                        // 1. The voxel must be a surface voxel
                        // 2. The voxel must be roughly tangent to the current surface
                        // 3. The magnitude of the voxel's normal vector must not be zero

                        point newP = {get_val(interp, newX, newY, newZ), newX, newY, newZ};
                        if (!newP.isSurface(interp)) continue;
                        
                        bool hasAlignedFace = false;
                        for (int i = 0; i < 6; i++){
                            if (!((newP.neighbors >> i) & 1) && referenceNormal.dot(directions6[i]) > 0.90f){
                                hasAlignedFace = true;
                                break;
                            }
                        }
                        
                        if (!hasAlignedFace) continue;

                        // This section is just checking if a voxel lies on the plane by checking that some of it's corners lie below and above the plane
                        cv::Vec3f voxelCorners[8];
                        cv::Vec3f origin(p.x, p.y, p.z);
                        int index = 0;

                        for (int dx = 0; dx <= 1; dx++)
                            for (int dy = 0; dy <= 1; dy++)
                                for (int dz = 0; dz <= 1; dz++)
                                    voxelCorners[index++] = cv::Vec3f(newX + dx, newY + dy, newZ + dz);

                        float fmin = 1000, fmax = -1000;
                        
                        for (int i = 0; i < 8; i++){
                            float f = p.normalizedNormal.dot(voxelCorners[i] - origin);
                            fmax = std::max(f, fmax);
                            fmin = std::min(f, fmin);
                        }

                        if (!(fmin <= 0 and fmax >= 0)) continue;

                        newP.computeNormal(interp);
                        if (cv::norm(newP.normal) == 0) continue; // Theoretically this should basically never happen but its here just in case
                        
                        // Project point onto plane, setting i and j
                        cv::Vec3f delta = cv::Vec3f(newX, newY, newZ) - globalOrigin;
                        newP.i = static_cast<int>(std::round(t1.dot(delta)));
                        newP.j = static_cast<int>(std::round(t2.dot(delta)));

                        if (!ij_points.insert((static_cast<int64_t>(newP.i) << 32) | (static_cast<uint32_t>(newP.j))).second) {
                            newP.ignore = true;
                        }

                        seen.insert(combineCoords(newX, newY, newZ));
                        q.push(std::make_tuple(std::make_shared<point>(newP), referenceNormal, t1, t2, patience, std::max(0, updateNormalTimer - 1)));
                    }
                }
            }
        }
    }

    //All done, hooray
    return;
}

inline void updateTangents(cv::Vec3f &oldNormal, cv::Vec3f &newNormal, cv::Vec3f &t1, cv::Vec3f &t2){
    cv::Vec3f rotationAxis = oldNormal.cross(newNormal);
    float sinTheta = cv::norm(rotationAxis);
    float cosTheta = oldNormal.dot(newNormal);

    if (sinTheta < 1e-5f) // If its basically the same normal just do nothing, no point risking division by 0
        return;

    rotationAxis /= sinTheta;
    
    auto rotate = [&](const cv::Vec3f &v) -> cv::Vec3f {
        return v * cosTheta +
               rotationAxis.cross(v) * sinTheta +
               rotationAxis * (rotationAxis.dot(v)) * (1 - cosTheta);
    };
    
    t1 = rotate(t1);
    t2 = rotate(t2);
    
    t1 -= newNormal * newNormal.dot(t1);
    t1 = cv::normalize(t1);
    t2 = cv::normalize(newNormal.cross(t1));

    return;
}

inline cv::Mat_<cv::Vec3f> downsampleGrid(const cv::Mat_<cv::Vec3f>& grid,int stepSize){
    int newH = grid.rows / stepSize;
    int newW = grid.cols / stepSize;
    int limit = stepSize * stepSize;

    cv::Mat_<cv::Vec3f> out(newH, newW, cv::Vec3f(-1,-1,-1));
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(stepSize / 2, stepSize / 12); // mean = stepSize / 2, std = stepSize / 12

    for (int i = 0; i + stepSize - 1 < grid.rows; i += stepSize){
        for (int j = 0; j + stepSize - 1 < grid.cols; j += stepSize){
            bool foundOne = false;
            int counter = 0;
                while (!foundOne && counter++ < limit){
                int di = i + static_cast<int>(std::round(dist(rng)));
                int dj = j + static_cast<int>(std::round(dist(rng)));

                while (di < i || di >= i + stepSize)
                    di = i + static_cast<int>(std::round(dist(rng)));
                while (dj < j || dj >= j + stepSize)
                    dj = j + static_cast<int>(std::round(dist(rng)));
                
                cv::Vec3f val = grid(di, dj);
                if (val != cv::Vec3f(-1,-1,-1)){
                    foundOne = true;
                    out[i / stepSize][j / stepSize] = val;
                }
            }
        }
    }





    return out;
}


// Done in 2 parts:
// First, fill first 3 layers of invalid cells with average of neighboring valid cells
// Second, fill rest of invalid cells with line of best fit
inline void fillHoles(cv::Mat_<cv::Vec3f>& grid, int invalidCellCount){
    // Averaging first 3 layers of invalid cells
    for (int iterations = 1; iterations <= 3 || invalidCellCount > 0; iterations++){
        cv::Mat_<cv::Vec3f> next = grid;
        for (int i = 0; i < grid.rows; i++){
            for (int j = 0; j < grid.cols; j++){
                if (grid(i, j) == cv::Vec3f(-1, -1, -1)){
                    int neighbors = 0;
                    float x_num = 0, y_num = 0, z_num = 0;

                    if (i - 1 >= 0){
                        cv::Vec3f neighbor = grid(i - 1, j);
                        if (neighbor != cv::Vec3f(-1, -1, -1)){
                            x_num += neighbor[0]; y_num += neighbor[1]; z_num += neighbor[2];
                            neighbors++;
                        }
                    }
                    if (j - 1 >= 0){
                        cv::Vec3f neighbor = grid(i, j - 1);
                        if (neighbor != cv::Vec3f(-1, -1, -1)){
                            x_num += neighbor[0]; y_num += neighbor[1]; z_num += neighbor[2];
                            neighbors++;
                        }
                    }
                    if (i + 1 < grid.rows){
                        cv::Vec3f neighbor = grid(i + 1, j);
                        if (neighbor != cv::Vec3f(-1, -1, -1)){
                            x_num += neighbor[0]; y_num += neighbor[1]; z_num += neighbor[2];
                            neighbors++;
                        }
                    }
                    if (j + 1 < grid.cols){
                        cv::Vec3f neighbor = grid(i, j + 1);
                        if (neighbor != cv::Vec3f(-1, -1, -1)){
                            x_num += neighbor[0]; y_num += neighbor[1]; z_num += neighbor[2];
                            neighbors++;
                        }
                    }

                    if (neighbors >= 2){ // neighbors acts as the denominator here
                        next(i, j) = cv::Vec3f(x_num / neighbors, y_num / neighbors, z_num / neighbors);
                        invalidCellCount--;
                    }
                }
            }
        }
        grid = next;
    }

    // Fill rest of invalid cells with line of best fit

}