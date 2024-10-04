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
    std::cerr << "Loaded " << model.source << " in " << timer.elapsed() << " s" << std::endl;
    SQDUMP(squash::bf16ToFloat(
        std::get<squash::tensor_data::BF16>(model.layers[5].mlp.up.data).data[0]));
    return 0;
}
