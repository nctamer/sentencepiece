// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "expansion_processor.h"
#include "init.h"

ABSL_FLAG(std::string, expansion, "", "serialized ExpansionResult");
ABSL_FLAG(std::string, hierarchy_file, "",
          "optional sentencepiece-bpe-hierarchy-v1 sidecar; required when "
          "the ExpansionResult uses hierarchical completion");
ABSL_FLAG(std::string, mode, "ids",
          "ids | pieces | spans | idmap_sha | idmap");

namespace {

using sentencepiece::expansion::CompletionGate;

bool LoadHierarchy(const std::string& path,
                   std::map<std::string, std::vector<CompletionGate>>* out,
                   std::string* error) {
  if (path.empty()) return true;
  std::ifstream in(path);
  if (!in) { *error = "cannot open hierarchy_file"; return false; }
  std::string line;
  if (!std::getline(in, line) || line != "# sentencepiece-bpe-hierarchy-v1") {
    *error = "bad hierarchy header"; return false;
  }
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) { *error = "bad hierarchy row"; return false; }
    const std::string surface = line.substr(0, tab);
    const std::string spec = line.substr(tab + 1);
    std::vector<CompletionGate> gates;
    if (!spec.empty()) {
      for (absl::string_view one : absl::StrSplit(spec, ';')) {
        const size_t colon = one.find(':');
        if (colon == absl::string_view::npos) {
          *error = "bad hierarchy gate"; return false;
        }
        CompletionGate gate;
        if (!absl::SimpleAtoi(one.substr(0, colon), &gate.level)) {
          *error = "bad hierarchy level"; return false;
        }
        for (absl::string_view cut :
             absl::StrSplit(one.substr(colon + 1), ',')) {
          int value = 0;
          if (!absl::SimpleAtoi(cut, &value)) {
            *error = "bad hierarchy cut"; return false;
          }
          gate.cuts.push_back(value);
        }
        gates.push_back(std::move(gate));
      }
    }
    if (!out->emplace(surface, std::move(gates)).second) {
      *error = "duplicate hierarchy surface"; return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  sentencepiece::expansion::ExpansionProcessor p;
  const auto st = p.LoadFromFile(absl::GetFlag(FLAGS_expansion));
  if (!st.ok()) { std::cerr << st.message() << "\n"; return 1; }

  std::map<std::string, std::vector<CompletionGate>> hierarchy;
  std::string hierarchy_error;
  const std::string hierarchy_path = absl::GetFlag(FLAGS_hierarchy_file);
  if (!LoadHierarchy(hierarchy_path, &hierarchy, &hierarchy_error)) {
    std::cerr << hierarchy_error << "\n"; return 1;
  }
  if (p.RequiresHierarchy() && hierarchy_path.empty()) {
    std::cerr << "hierarchical expansion requires --hierarchy_file\n";
    return 1;
  }

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
    absl::Status e;
    if (p.RequiresHierarchy()) {
      const std::string normalized = p.Normalize(line);
      const auto it = hierarchy.find(normalized);
      if (it == hierarchy.end()) {
        std::cerr << "hierarchy_file has no normalized row for input: "
                  << normalized << "\n";
        return 1;
      }
      e = p.EncodeWithHierarchy(line, it->second, &spans);
    } else {
      e = p.Encode(line, &spans);
    }
    if (!e.ok()) { std::cerr << e.message() << "\n"; return 1; }

    std::vector<std::string> out;
    for (const auto& span : spans) {
      if (mode == "ids") out.push_back(std::to_string(span.id));
      else if (mode == "pieces") out.push_back(span.piece);
      else if (mode == "spans") {
        out.push_back(std::to_string(span.begin) + ":" +
                      std::to_string(span.end));
      } else {
        std::cerr << "unknown mode: " << mode << "\n"; return 1;
      }
    }
    std::cout << absl::StrJoin(out, "\t") << "\n";
  }
  return 0;
}
