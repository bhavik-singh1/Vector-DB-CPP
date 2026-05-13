#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

// Ensure strict alignment for cross-platform/binary compatibility
#pragma pack(push, 1)
struct NebulaSuperblock {
    char     magic[8];          // "NEBULA\0\0"
    uint32_t version;
    uint32_t flags;
    uint64_t timestamp;
    uint32_t vector_dim;
    uint32_t distance_metric;
    uint64_t total_records;
    uint64_t next_record_id;
    uint32_t page_size;
    uint32_t hnsw_M;
    uint32_t hnsw_ef_const;
    uint64_t hnsw_entry_node;
    int32_t  hnsw_max_layer;
    uint8_t  padding[3996];     // Pad to 4092 bytes
    uint32_t checksum;
};

struct RecordHeader {
    uint64_t id;
    uint64_t metadata_offset;
    uint64_t graph_offset;
    uint32_t metadata_len;
    uint32_t neighbor_count;
    uint8_t  flags;
    uint8_t  padding[7];
};
#pragma pack(pop)

enum class WALOp : uint8_t {
    OP_INSERT = 0,
    OP_DELETE = 1,
    OP_UPDATE_GRAPH = 2
};

#pragma pack(push, 1)
struct WALHeader {
    uint32_t crc32;      // Checksum of the payload
    uint32_t payloadLen; // Size of the data following this header
    uint64_t lsn;        // Log Sequence Number
    uint8_t  op;         // WALOp
};
#pragma pack(pop)

class WALManager {
public:
    WALManager();
    ~WALManager();
    void open(const std::string& logPath);
    void append(WALOp op, uint64_t id, const void* data, size_t len);
    void truncate();
private:
    HANDLE hLog = INVALID_HANDLE_VALUE;
    uint64_t nextLsn = 0;
    uint32_t calculateCRC32(const void* data, size_t len);
};

class StorageManager {
public:
    struct Config {
        uint32_t dim;
        uint32_t metric;
        uint32_t hnsw_M = 16;
        uint32_t hnsw_ef_const = 200;
    };

    StorageManager();
    ~StorageManager();

    void initialize(const std::string& basePath, const Config& cfg);
    void close();

    uint64_t insert_record(const std::vector<float>& vec, const std::string& metadata);
    void flush_superblock();

    uint64_t get_record_count() const { return sb.total_records; }
    void get_record(uint64_t id, std::vector<float>& vec, std::string& metadata);

private:
    std::string basePath;
    NebulaSuperblock sb;
    size_t recordStride;
    const size_t SUPERBLOCK_PADDING = 65536;

    HANDLE hRecords = INVALID_HANDLE_VALUE;
    HANDLE hMetadata = INVALID_HANDLE_VALUE;
    HANDLE hGraph = INVALID_HANDLE_VALUE;
    HANDLE hMapRecords = NULL;
    void*  pRecordsBase = nullptr;

    WALManager wal;

    HANDLE openFile(const std::string& suffix);
    void setupNewDatabase(const Config& cfg);
    void loadExistingDatabase();
    void recover_from_wal();
    void ensure_capacity(uint64_t record_count);
    float* get_vector(uint64_t internal_id);
    RecordHeader* get_header(uint64_t internal_id);
};
