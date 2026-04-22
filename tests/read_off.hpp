#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct OffData
{
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::vector<std::size_t>> faces;
};

inline OffData
read_off(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    // Skip blank lines and comments, then read the OFF header
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.substr(0, 3) != "OFF") {
            throw std::runtime_error("Invalid OFF file: missing OFF header");
        }
        break;
    }

    // Read counts (may be on the same line as OFF or on the next line)
    std::size_t n_vertices = 0, n_faces = 0, n_edges = 0;
    std::string rest = line.substr(3);
    std::istringstream count_stream(rest);
    if (!(count_stream >> n_vertices >> n_faces >> n_edges)) {
        // Counts are on the next non-empty line
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            if (!(iss >> n_vertices >> n_faces >> n_edges)) {
                throw std::runtime_error("Invalid OFF file: cannot read vertex/face/edge counts");
            }
            break;
        }
    }

    OffData data;
    data.vertices.reserve(n_vertices);
    data.faces.reserve(n_faces);

    // Read vertices
    for (std::size_t i = 0; i < n_vertices; ++i) {
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            break;
        }
        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
            throw std::runtime_error("Invalid OFF file: cannot read vertex " + std::to_string(i));
        }
        data.vertices.push_back({ x, y, z });
    }

    // Read faces
    for (std::size_t i = 0; i < n_faces; ++i) {
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            break;
        }
        std::istringstream iss(line);
        std::size_t n_verts;
        if (!(iss >> n_verts)) {
            throw std::runtime_error("Invalid OFF file: cannot read face " + std::to_string(i));
        }
        std::vector<std::size_t> face(n_verts);
        for (std::size_t j = 0; j < n_verts; ++j) {
            if (!(iss >> face[j])) {
                throw std::runtime_error("Invalid OFF file: cannot read vertex index " + std::to_string(j) +
                                         " of face " + std::to_string(i));
            }
        }
        data.faces.push_back(std::move(face));
    }

    return data;
}
