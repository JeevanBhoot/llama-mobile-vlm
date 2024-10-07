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
    auto model = squash::sqt_load(modelFile);
    auto generator = squash::Generator(model);
    std::cerr << "Loaded " << model.source << " (" << timer.elapsed() << " s)\n";

    auto nSteps = 4u;
    timer = squash::Timer();
    std::vector<uint> tokens({128000, 791, 7438, 315, 2324, 374});
    tokens.push_back(generator.prefill(tokens, nSteps));
    std::cerr << "Prefill (" << double(tokens.size() - 1) / timer.elapsed() << " token/s)\n";

    timer = squash::Timer();
    for (auto i = 0u; i < nSteps; ++i) {
        tokens.push_back(generator.generate());
    }
    std::cerr << "Generation (" << nSteps / timer.elapsed() << " token/s)\n";
    std::cerr << squash::dump(tokens) << std::endl;

    return 0;
}
