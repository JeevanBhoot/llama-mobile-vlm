#include "benchmark/benchmark.hpp"
#include <cxxopts.hpp>

int main(int argc, char** argv) {
    using strings = std::vector<std::string>;
    cxxopts::Options options("benchmark", "Benchmarks");
    options.add_options()                                                                    //
        ("prefix", "Prefixes to select tests", cxxopts::value<strings>())                    //
        ("j,json", "Output JSON to stdout", cxxopts::value<bool>()->default_value("false"))  //
        ("r,repeat", "Number of times to repeat each benchmark",                             //
         cxxopts::value<uint>()->default_value("1"))                                         //
        ("s,shuffle", "Shuffle benchmark and case order",                                    //
         cxxopts::value<bool>()->default_value("false"))                                     //
        ("help", "Print help")                                                               //
        ;
    options.parse_positional({"prefix"});
    options.positional_help("prefix...");
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cerr << options.help() << std::endl;
        return 0;
    }
    squash::benchmarking::Registry::run(
        args.count("prefix") ? args["prefix"].as<strings>() : strings{}, args["json"].as<bool>(),
        args["repeat"].as<uint>(), args["shuffle"].as<bool>());
    return 0;
}
