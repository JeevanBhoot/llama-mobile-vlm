#include <cxxopts.hpp>
#include <fstream>
#include <iostream>

#include "lib/squash.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("cli", "Text generation CLI");
    options.add_options()                                                      //
        ("model_file", "Model to load (.sqt)", cxxopts::value<std::string>())  //
        ("image", "Image file", cxxopts::value<std::string>())                 //
        ("help", "Print help")                                                 //
        ("g,max_generated_tokens", "Maximum number of generated tokens",
         cxxopts::value<uint>()->default_value("16"))                                           //
        ("t,temperature", "Sampling temperature", cxxopts::value<float>()->default_value("0"))  //
        ("top_k", "Sampling top-k", cxxopts::value<uint>()->default_value("50"))                //
        ("top_p", "Sampling top-p", cxxopts::value<float>()->default_value("1"))                //
        ;
    options.parse_positional({"model_file"});
    options.positional_help("model_file");
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cerr << options.help() << std::endl;
        return 0;
    }

    squash::selectOmpNumThreads();

    squash::Timer timer;
    std::ifstream modelFile(args["model_file"].as<std::string>());
    auto model = squash::loadSquashedTensors(modelFile);
    auto generator = squash::Generator(model);
    std::cerr << "-- Loaded " << model.source << " (" << timer.elapsed() << " s)\n\n";

    std::optional<squash::Image> image;
    if (args.count("image")) {
        image.emplace(squash::loadImage(args["image"].as<std::string>()));
    }

    std::string prompt;
    squash::Generator::Options generatorOptions{
        .maxGeneratedTokens = args["max_generated_tokens"].as<uint>(),
        .seed = std::nullopt,
        .temperature = args["temperature"].as<float>(),
        .topK = args["top_k"].as<uint>(),
        .topP = args["top_p"].as<float>(),
    };
    while (std::getline(std::cin, prompt)) {
        timer = squash::Timer();
        auto prefillOut = generator.prefill(prompt, image, generatorOptions);
        std::cout << prefillOut.back();
        auto prefillTime = timer.elapsed();
        timer = squash::Timer();
        auto step = 0u;
        while (true) {
            auto next = generator.generate();
            std::cout << next << std::flush;
            ++step;
            if (next.empty()) {
                break;
            }
        }
        std::cout << "\n";
        auto generateRate = step / timer.elapsed();
        std::cerr << "-- Prefill (" << prefillOut.size() << " tok) " << prefillTime
              << " s; Generate " << generateRate << " tok/s\n\n";
    }
    return 0;
}
