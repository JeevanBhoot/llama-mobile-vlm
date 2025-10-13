#include "benchmark/benchmark.hpp"
#include <cxxopts.hpp>

int main(int argc, char** argv) {
    cxxopts::Options options("benchmark", "Benchmarks");
    options.add_options()                                                                       //
        ("prefix", "Prefix to select tests", cxxopts::value<std::string>()->default_value(""))  //
        ("json", "Output JSON to stdout", cxxopts::value<bool>()->default_value("false"))       //
        ("help", "Print help")                                                                  //
        ;
    options.parse_positional({"prefix"});
    options.positional_help("prefix");
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cerr << options.help() << std::endl;
        return 0;
    }
    squash::benchmarking::Registry::run(args["prefix"].as<std::string>(), args["json"].as<bool>());
    return 0;
}
