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
    std::cerr << "Loaded " << model.source << " in " << timer.elapsed() << " s\n";
    timer = squash::Timer();
    auto next = generator.prefill({128000, 791, 7438, 374}, 1u);
    std::cerr << "Prefill in " << timer.elapsed() << " s\n";
    std::cerr << " -> " << next << "\n";
    return 0;
}
