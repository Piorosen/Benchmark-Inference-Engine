#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <argparse/argparse.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stbi_image.h"

int main(int argc, char** argv) { 
    argparse::ArgumentParser program("EventNVR");

    program.add_argument("-p")
        .help("web server port").required()
        .default_value("./model.param");

    program.add_argument("-b")
        .help("web server port").required()
        .default_value("./model.bin");
    
    program.add_argument("-i")
        .help("web server port").required()
        .default_value("./image.png");

    program.add_argument("-s")
        .default_value(224).required()
        .help("web server port")
        .scan<'i', int>();
    
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << program << '\n';
        std::exit(1);
    }

    std::string param_path = program.get("-p");
    std::string bin_path = program.get("-b");
    std::string img_path = program.get("-i");
    int image_size = program.get<int>("-s");

    std::cout << "param : " << param_path << std::endl;
    std::cout << "bin_path : " << bin_path << std::endl;
    std::cout << "img_path : " << img_path << std::endl;
    std::cout << "image_size : " << image_size << std::endl;
    
    ncnn::Net detector;
    // CPU, GPU 플래그
    
    if (detector.load_param(param_path.c_str()) != 0) {
        return false;
    }
    if (detector.load_model(bin_path.c_str()) != 0) {
        return false;
    }

    // stbi_image_free()
    // cv::Mat image = cv::imread(img_path);
    int image_w, image_h;
    unsigned char* data = stbi_load(img_path.c_str(), &image_w, &image_h, nullptr, 0);
    int w = image_w; int h = image_h; float scale = 1.f;

    if (w > h) {
        scale = (float) image_size / w;
        w = image_size;
        h = h * scale;
    } else {
        scale = (float) image_size / h;
        h = image_size;
        w = w  * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(data, ncnn::Mat::PIXEL_BGR, image_w, image_h, w, h);
    int w_pad = image_size - w;
    int h_pad = image_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, h_pad, 0, w_pad, ncnn::BORDER_CONSTANT, 114.f);

    std::vector<std::chrono::nanoseconds> time;

    for (int i = 0; i < 50; i++) { 
        auto start = std::chrono::high_resolution_clock::now();
        detector.opt.use_vulkan_compute = false;
        ncnn::Extractor ex = detector.create_extractor();

        ncnn::Mat input;
        ncnn::Mat output;
        ex.input("input", in);
        ex.extract("output", output);
        // ex.extract("masks", output);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "[ " << i + 1 << "/" << 50 << "]\t" << std::chrono::duration_cast<std::chrono::milliseconds>((end - start)).count() << "ms" << std::endl;
        time.push_back((end - start));
    }



    return 0;
}