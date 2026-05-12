#include "StorageManager.h"
#include <iostream>

// =====================================================================
// WALManager Implementation
// =====================================================================

WALManager::WALManager() {}
WALManager::~WALManager() { if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog); }

void WALManager::open(const std::string& logPath) {
    hLog = CreateFileA(logPath.c_str(), GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_FLAG_WRITE_THROUGH, NULL);
    if (hLog == INVALID_HANDLE_VALUE) throw std::runtime_error("Could not open WAL log");
    SetFilePointer(hLog, 0, NULL, FILE_END);
}

void WALManager::append(WALOp op, uint64_t id, const void* data, size_t len) {
    WALHeader header;
    header.op = (uint8_t)op;
    header.lsn = nextLsn++;
    header.payloadLen = (uint32_t)len;
    header.crc32 = calculateCRC32(data, len);
    DWORD written;
    WriteFile(hLog, &header, sizeof(header), &written, NULL);
    WriteFile(hLog, data, (DWORD)len, &written, NULL);
}

void WALManager::truncate() {
    SetFilePointer(hLog, 0, NULL, FILE_BEGIN);
    SetEndOfFile(hLog);
    nextLsn = 0;
}

uint32_t WALManager::calculateCRC32(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(int)(crc & 1)));
        }
    }
    return ~crc;
}

// =====================================================================
// StorageManager Implementation
// =====================================================================

StorageManager::StorageManager() {}
StorageManager::~StorageManager() { close(); }

HANDLE StorageManager::openFile(const std::string& suffix) {
    std::string path = basePath + suffix;
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open file: " + path);
    return h;
}

void StorageManager::initialize(const std::string& path, const Config& cfg) {
    this->basePath = path;
    hRecords = openFile(".records.dat");
    hMetadata = openFile(".metadata.dat");
    hGraph = openFile(".graph.dat");

    DWORD fileSizeHigh;
    DWORD fileSizeLow = GetFileSize(hRecords, &fileSizeHigh);
    
    if (fileSizeLow == 0 && fileSizeHigh == 0) {
        setupNewDatabase(cfg);
    } else {
        loadExistingDatabase();
    }

    recordStride = sizeof(RecordHeader) + (sb.vector_dim * sizeof(float));
    size_t mapSize = SUPERBLOCK_PADDING + (sb.total_records * recordStride) + (1024 * 1024 * 10);

    hMapRecords = CreateFileMappingA(hRecords, NULL, PAGE_READWRITE, 0, (DWORD)mapSize, NULL);
    if (!hMapRecords) throw std::runtime_error("Mapping failed");

    pRecordsBase = MapViewOfFile(hMapRecords, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!pRecordsBase) throw std::runtime_error("View mapping failed");

    wal.open(path + ".wal");
    if (sb.flags & 0x01) recover_from_wal();

    std::cout << "NebulaDB Storage Initialized. Records: " << sb.total_records << " | Dim: " << sb.vector_dim << std::endl;
}

void StorageManager::setupNewDatabase(const Config& cfg) {
    memset(&sb, 0, sizeof(sb));
    memcpy(sb.magic, "NEBULA\0\0", 8);
    sb.version = 1;
    sb.flags = 0x01; // Set dirty bit during setup
    sb.vector_dim = cfg.dim;
    sb.distance_metric = cfg.metric;
    sb.page_size = 4096;
    sb.hnsw_M = cfg.hnsw_M;
    sb.hnsw_ef_const = cfg.hnsw_ef_const;
    sb.total_records = 0;
    
    DWORD written;
    WriteFile(hRecords, &sb, sizeof(sb), &written, NULL);
    SetFilePointer(hRecords, SUPERBLOCK_PADDING, NULL, FILE_BEGIN);
    SetEndOfFile(hRecords);
}

void StorageManager::loadExistingDatabase() {
    DWORD read;
    SetFilePointer(hRecords, 0, NULL, FILE_BEGIN);
    if (!ReadFile(hRecords, &sb, sizeof(sb), &read, NULL)) throw std::runtime_error("Failed to read superblock");
    if (memcmp(sb.magic, "NEBULA\0\0", 8) != 0) throw std::runtime_error("Invalid database format");
}

void StorageManager::recover_from_wal() {
    // To be implemented: Scan WAL and apply missing updates
    std::cout << "Recovery required (Not yet implemented)" << std::endl;
}

void StorageManager::close() {
    flush_superblock();
    if (pRecordsBase) UnmapViewOfFile(pRecordsBase);
    if (hMapRecords) CloseHandle(hMapRecords);
    if (hRecords != INVALID_HANDLE_VALUE) CloseHandle(hRecords);
    if (hMetadata != INVALID_HANDLE_VALUE) CloseHandle(hMetadata);
    if (hGraph != INVALID_HANDLE_VALUE) CloseHandle(hGraph);
    
    pRecordsBase = nullptr;
    hMapRecords = NULL;
    hRecords = hMetadata = hGraph = INVALID_HANDLE_VALUE;
}

uint64_t StorageManager::insert_record(const std::vector<float>& vec, const std::string& metadata) {
    if (vec.size() != sb.vector_dim) throw std::runtime_error("Vector dimension mismatch");

    uint64_t id = sb.next_record_id++;
    ensure_capacity(sb.next_record_id);

    // 1. Write to WAL first (Durability)
    // For simplicity, we log the vector. In production, log metadata too.
    wal.append(WALOp::INSERT, id, vec.data(), vec.size() * sizeof(float));

    // 2. Update Metadata File (Append-only)
    DWORD bytesWritten;
    uint64_t metaOffset = GetFileSize(hMetadata, NULL);
    uint32_t metaLen = (uint32_t)metadata.size();
    WriteFile(hMetadata, metadata.data(), metaLen, &bytesWritten, NULL);

    // 3. Update Memory-Mapped Record
    RecordHeader* header = get_header(id);
    header->id = id;
    header->metadata_offset = metaOffset;
    header->metadata_len = metaLen;
    header->neighbor_count = 0;
    header->flags = 0;

    float* vecPtr = get_vector(id);
    memcpy(vecPtr, vec.data(), vec.size() * sizeof(float));

    sb.total_records++;
    
    // Periodically flush superblock to disk
    if (sb.total_records % 100 == 0) flush_superblock();

    return id;
}

void StorageManager::ensure_capacity(uint64_t record_count) {
    size_t neededSize = SUPERBLOCK_PADDING + (record_count * recordStride);
    
    // Get current mapping size (simple version: check file size)
    DWORD high;
    size_t currentFileSize = GetFileSize(hRecords, &high);

    if (neededSize > currentFileSize) {
        // We need to grow the file and remap
        // In a production system, we'd over-allocate (e.g., 2x growth)
        size_t newSize = neededSize + (1024 * 1024 * 10); // +10MB cushion

        UnmapViewOfFile(pRecordsBase);
        CloseHandle(hMapRecords);

        SetFilePointer(hRecords, (LONG)newSize, NULL, FILE_BEGIN);
        SetEndOfFile(hRecords);

        hMapRecords = CreateFileMappingA(hRecords, NULL, PAGE_READWRITE, 0, (DWORD)newSize, NULL);
        pRecordsBase = MapViewOfFile(hMapRecords, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    }
}

void StorageManager::flush_superblock() {
    DWORD written;
    SetFilePointer(hRecords, 0, NULL, FILE_BEGIN);
    WriteFile(hRecords, &sb, sizeof(sb), &written, NULL);
}

float* StorageManager::get_vector(uint64_t internal_id) {
    uint8_t* ptr = (uint8_t*)pRecordsBase + SUPERBLOCK_PADDING + (internal_id * recordStride);
    return (float*)(ptr + sizeof(RecordHeader));
}

RecordHeader* StorageManager::get_header(uint64_t internal_id) {
    uint8_t* ptr = (uint8_t*)pRecordsBase + SUPERBLOCK_PADDING + (internal_id * recordStride);
    return (RecordHeader*)ptr;
}
