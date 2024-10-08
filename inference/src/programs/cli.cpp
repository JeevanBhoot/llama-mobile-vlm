#include <fstream>
#include <iostream>
#include "squash.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Error - no model specified."
                  << "\nUsage: ./demo path/to/model.sqt" << std::endl;
        return 1;
    }
    std::ifstream modelFile(argv[1]);
    squash::Timer timer;
    auto model = squash::loadSquashedTensors(modelFile);
    auto generator = squash::Generator(model);
    std::cerr << "-- Loaded " << model.source << " (" << timer.elapsed() << " s)\n";

    auto nSteps = 4u;
    timer = squash::Timer();
    std::string prompt = "The meaning of life is";
    auto completion = generator.prefill(prompt, nSteps);
    std::cerr << "-- Prefill (" << double(generator.prefillLength) / timer.elapsed()
              << " token/s)\n";

    timer = squash::Timer();
    for (auto i = 0u; i < nSteps; ++i) {
        completion += generator.generate();
    }
    std::cerr << "-- Generation (" << nSteps / timer.elapsed() << " token/s)\n";
    std::cerr << prompt + "|" + completion << std::endl;

    return 0;
}
