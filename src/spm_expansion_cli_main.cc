// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Reference CLI for the authoritative explicit-program BPE runtime.
// One line of input -> one line of output, so another language binding can be
// checked against the C++ semantics instead of reimplementing them blind.

#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/str_join.h"
#include "expansion_processor.h"
#include "init.h"

ABSL_FLAG(std::string, expansion, "", "serialized ExpansionResult");
ABSL_FLAG(std::string, mode, "ids",
          "ids | pieces | spans | idmap_sha | idmap (the <id>\\t<piece>\\t<type> "
          "preimage of idmap_sha)");

int main(int argc, char* argv[]) {
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  sentencepiece::expansion::ExpansionProcessor p;
  const auto st = p.LoadFromFile(absl::GetFlag(FLAGS_expansion));
  if (!st.ok()) { std::cerr << st.message() << "\n"; return 1; }
  const std::string mode = absl::GetFlag(FLAGS_mode);
  if (mode == "idmap_sha") { std::cout << p.IdMapSha256() << "\n"; return 0; }
  if (mode == "idmap") {
    for (int i = 0; i < p.GetPieceSize(); ++i) {
      std::cout << i << "\t" << p.IdToPiece(i) << "\t" << p.IdToType(i) << "\n";
    }
    return 0;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<sentencepiece::expansion::TokenSpan> spans;
    const auto e = p.Encode(line, &spans);
    if (!e.ok()) { std::cerr << e.message() << "\n"; return 1; }
    std::vector<std::string> out;
    for (const auto& s : spans) {
      if (mode == "ids") out.push_back(std::to_string(s.id));
      else if (mode == "pieces") out.push_back(s.piece);
      else out.push_back(std::to_string(s.begin) + ":" + std::to_string(s.end));
    }
    std::cout << absl::StrJoin(out, "\t") << "\n";
  }
  return 0;
}
