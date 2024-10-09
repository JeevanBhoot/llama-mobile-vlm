#include <fstream>
#include <iostream>
#include "squash.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Error - no model specified."
                  << "\nUsage: ./demo path/to/model.sqt [max_gen_tokens]" << std::endl;
        return 1;
    }
    std::ifstream modelFile(argv[1]);
    auto nSteps = (argc < 3) ? 16u : uint(std::atoi(argv[2]));

    squash::Timer timer;
    auto model = squash::loadSquashedTensors(modelFile);
    auto generator = squash::Generator(model);
    std::cerr << "-- Loaded " << model.source << " (" << timer.elapsed() << " s)\n\n";

    std::string prompt;
    while (std::getline(std::cin, prompt)) {
        timer = squash::Timer();
        std::cout << generator.prefill(prompt, nSteps);
        auto prefillRate = generator.prefillLength / timer.elapsed();
        timer = squash::Timer();
        for (auto i = 0u; i < nSteps; ++i) {
            auto next = generator.generate();
            std::cout << next << std::flush;
            if (next.empty()) {
                break;
            }
        }
        std::cout << "\n";
        auto generateRate = generator.generateLength / timer.elapsed();
        std::cerr << "-- Prefill " << prefillRate << " tok/s; Generate " << generateRate
                  << " tok/s\n\n";
    }
    return 0;
}
