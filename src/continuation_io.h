// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef CONTINUATION_IO_H_
#define CONTINUATION_IO_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"

namespace sentencepiece::continuation {

using Sentence = std::pair<std::string, int64_t>;

struct PreparedCorpus {
  std::vector<Sentence> sentences;
  int64_t weighted_sentence_count = 0;
};

// Continuation deliberately treats one input record as one hard training
// domain. It therefore does not use TrainerInterface::LoadSentences(), whose
// character-coverage replacement and legacy pretokenizer facilities are fresh
// training policies rather than inherited-tokenizer state.
absl::Status LoadPreparedCorpus(const TrainerSpec& trainer_spec,
                                const NormalizerSpec& normalizer_spec,
                                const TrainerComponents& components,
                                PreparedCorpus* corpus);

absl::Status ReadExpansionSpec(absl::string_view filename,
                               ExpansionSpec* spec);
absl::Status ReadModelProto(absl::string_view filename, ModelProto* model,
                            std::string* raw_bytes = nullptr);
absl::Status WriteExpansionResult(absl::string_view filename,
                                  const ExpansionResult& result);
absl::Status WriteModelProto(absl::string_view filename,
                             const ModelProto& model);
absl::Status WriteExpansionVocab(
    absl::string_view filename,
    const std::vector<ExpansionPiece>& externally_ordered_pieces);
absl::Status WriteMergeTable(
    absl::string_view filename,
    const std::vector<ExpansionMerge>& rank_ordered_merges);

// SHA-256 over exact bytes, lowercase hexadecimal.
std::string Sha256Hex(absl::string_view bytes);

}  // namespace sentencepiece::continuation

#endif  // CONTINUATION_IO_H_
