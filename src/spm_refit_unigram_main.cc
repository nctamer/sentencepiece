// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Fixed-vocabulary Unigram score refit.
//
// A DEDICATED executable on purpose. Routing this through spm_train would put
// vocab_size, seed pieces, character coverage and pruning within one typo of
// an operation whose entire contract is that support cannot change. Here there
// is no vocab_size flag to set, so support mutation is not expressible.

#include <cstdint>
#include <string>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "init.h"
#include "sentencepiece_model.pb.h"
#include "unigram_refit.h"
#include "util.h"

ABSL_FLAG(std::string, model, "", "input UNIGRAM .model whose support is kept");
ABSL_FLAG(std::string, input, "", "comma separated weighted corpus files");
ABSL_FLAG(std::string, input_format, "tsv", "text or tsv (<record><TAB><count>)");
ABSL_FLAG(std::string, model_prefix, "", "output model prefix");
ABSL_FLAG(int32_t, num_iterations, 8, "fixed-support EM iterations");
ABSL_FLAG(int32_t, num_threads, 1, "E-step threads");
ABSL_FLAG(double, count_floor, 1e-30,
          "numerical floor on posterior counts; never deletes a piece");
ABSL_FLAG(double, objective_tolerance, 0.0,
          "deterministic early stop when the objective improves by less than "
          "this; <= 0 disables");
ABSL_FLAG(int32_t, max_sentence_length, 8192,
          "refuse rather than silently drop a longer record");

int main(int argc, char* argv[]) {
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  QCHECK(!absl::GetFlag(FLAGS_model).empty()) << "--model is required";
  QCHECK(!absl::GetFlag(FLAGS_input).empty()) << "--input is required";
  QCHECK(!absl::GetFlag(FLAGS_model_prefix).empty())
      << "--model_prefix is required";

  sentencepiece::TrainerSpec corpus_spec;
  for (const auto& f :
       sentencepiece::util::StrSplitAsCSV(absl::GetFlag(FLAGS_input))) {
    corpus_spec.add_input(f);
  }
  corpus_spec.set_input_format(absl::GetFlag(FLAGS_input_format));
  corpus_spec.set_max_sentence_length(absl::GetFlag(FLAGS_max_sentence_length));

  sentencepiece::unigram_refit::RefitOptions options;
  options.num_iterations = absl::GetFlag(FLAGS_num_iterations);
  options.num_threads = absl::GetFlag(FLAGS_num_threads);
  options.count_floor = absl::GetFlag(FLAGS_count_floor);
  options.objective_tolerance = absl::GetFlag(FLAGS_objective_tolerance);

  QCHECK_OK(sentencepiece::unigram_refit::RefitFixedVocabularyToFiles(
      absl::GetFlag(FLAGS_model), corpus_spec, options,
      absl::GetFlag(FLAGS_model_prefix)));
  return 0;
}
