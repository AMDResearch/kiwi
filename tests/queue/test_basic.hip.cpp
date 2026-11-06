// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_basic.hip.cpp
 * @brief Basic unit tests for queue (CPU-only)
 *
 * Tests wave-slot packing/unpacking, epoch handling, and configuration.
 */

#include <kiwi/queue/queue.hpp>
#include <gtest/gtest.h>
#include <cstring>

using namespace kiwi::queue;

// Helper: simulate wave-slot packing by calling pack_wave_lane for all 8 lanes
static void pack_all_lanes(const void* data, uint64_t slot[8], uint8_t epoch, int wave_slot_id = 0) {
    for (int lane = 0; lane < 8; ++lane) {
        slot[lane] = pack_wave_lane(data, lane, epoch, wave_slot_id);
    }
}

// ============================================================
// Wave-slot packing and unpacking tests
// ============================================================

TEST(SlotTest, PackUnpack_uint32) {
    // Need 56-byte zero-padded buffer for pack_wave_lane
    uint8_t padded[56] = {};
    uint32_t original = 0x12345678;
    std::memcpy(padded, &original, sizeof(original));
    uint8_t epoch = 42;

    uint64_t slot[8];
    pack_all_lanes(padded, slot, epoch);

    // Verify epoch in lane 7
    EXPECT_EQ(get_epoch(slot[7]), epoch);

    // Unpack
    uint8_t buf[56] = {};
    unpack_wave_slot(slot, buf);
    uint32_t unpacked = 0;
    std::memcpy(&unpacked, buf, sizeof(unpacked));
    EXPECT_EQ(unpacked, original);
}

TEST(SlotTest, PackUnpack_uint16) {
    uint8_t padded[56] = {};
    uint16_t original = 0xABCD;
    std::memcpy(padded, &original, sizeof(original));
    uint8_t epoch = 128;

    uint64_t slot[8];
    pack_all_lanes(padded, slot, epoch);

    EXPECT_EQ(get_epoch(slot[7]), epoch);

    uint8_t buf[56] = {};
    unpack_wave_slot(slot, buf);
    uint16_t unpacked = 0;
    std::memcpy(&unpacked, buf, sizeof(unpacked));
    EXPECT_EQ(unpacked, original);
}

TEST(SlotTest, PackUnpack_uint8) {
    uint8_t padded[56] = {};
    uint8_t original = 0xFF;
    padded[0] = original;
    uint8_t epoch = 0;

    uint64_t slot[8];
    pack_all_lanes(padded, slot, epoch);

    EXPECT_EQ(get_epoch(slot[7]), epoch);

    uint8_t buf[56] = {};
    unpack_wave_slot(slot, buf);
    EXPECT_EQ(buf[0], original);
}

// Test epoch wrapping (0-255) with wave-slot packing
TEST(SlotTest, EpochWrapping) {
    uint8_t padded[56] = {};
    uint32_t data = 0xDEADBEEF;
    std::memcpy(padded, &data, sizeof(data));

    for (int i = 0; i < 300; ++i) {
        uint8_t epoch = static_cast<uint8_t>(i & 0xFF);
        uint64_t slot[8];
        pack_all_lanes(padded, slot, epoch);

        EXPECT_EQ(get_epoch(slot[7]), epoch);

        uint8_t buf[56] = {};
        unpack_wave_slot(slot, buf);
        uint32_t unpacked = 0;
        std::memcpy(&unpacked, buf, sizeof(unpacked));
        EXPECT_EQ(unpacked, data);
    }
}

// Test struct packing (small struct in lane 0)
struct TestStructSmall {
    uint32_t a;
    uint16_t b;
    uint8_t c;
} __attribute__((packed));

static_assert(sizeof(TestStructSmall) == 7, "TestStructSmall must be 7 bytes");

TEST(SlotTest, PackUnpack_SmallStruct) {
    uint8_t padded[56] = {};
    TestStructSmall original = {0x11223344, 0x5566, 0x77};
    std::memcpy(padded, &original, sizeof(original));
    uint8_t epoch = 200;

    uint64_t slot[8];
    pack_all_lanes(padded, slot, epoch);

    EXPECT_EQ(get_epoch(slot[7]), epoch);

    uint8_t buf[56] = {};
    unpack_wave_slot(slot, buf);
    TestStructSmall unpacked{};
    std::memcpy(&unpacked, buf, sizeof(unpacked));
    EXPECT_EQ(unpacked.a, original.a);
    EXPECT_EQ(unpacked.b, original.b);
    EXPECT_EQ(unpacked.c, original.c);
}

// ============================================================
// 56-byte maximum payload test
// ============================================================

TEST(SlotTest, PackUnpack_MaxPayload56) {
    uint8_t original[56];
    // Fill with a recognizable pattern
    for (int i = 0; i < 56; ++i) {
        original[i] = static_cast<uint8_t>(i + 1);
    }

    uint8_t epoch = 255;

    uint64_t slot[8];
    pack_all_lanes(original, slot, epoch);

    EXPECT_EQ(get_epoch(slot[7]), epoch);

    uint8_t unpacked[56] = {};
    unpack_wave_slot(slot, unpacked);
    for (int i = 0; i < 56; ++i) {
        EXPECT_EQ(unpacked[i], original[i])
            << "Mismatch at byte " << i;
    }
}

TEST(SlotTest, PackUnpack_MaxPayload56_AllFF) {
    uint8_t original[56];
    std::memset(original, 0xFF, 56);

    uint8_t epoch = 0;

    uint64_t slot[8];
    pack_all_lanes(original, slot, epoch);

    EXPECT_EQ(get_epoch(slot[7]), epoch);

    uint8_t unpacked[56] = {};
    unpack_wave_slot(slot, unpacked);
    for (int i = 0; i < 56; ++i) {
        EXPECT_EQ(unpacked[i], 0xFF)
            << "Mismatch at byte " << i;
    }
}

// ============================================================
// Epoch extraction from lane 7
// ============================================================

TEST(SlotTest, EpochExtraction_Lane7) {
    uint8_t padded[56] = {};
    uint32_t data = 0xCAFEBABE;
    std::memcpy(padded, &data, sizeof(data));

    // Verify epoch is correctly placed in lane 7 as full uint64_t
    for (uint16_t e = 0; e <= 255; ++e) {
        uint8_t epoch = static_cast<uint8_t>(e);
        uint64_t lane7_val = pack_wave_lane(padded, 7, epoch);

        // Extract epoch via get_epoch
        EXPECT_EQ(get_epoch(lane7_val), epoch);

        // Verify lane 7 value is just the epoch cast to uint64_t
        EXPECT_EQ(lane7_val, static_cast<uint64_t>(epoch));
    }
}

TEST(SlotTest, EpochDoesNotCorruptPayload) {
    // Verify that lanes 0-6 do NOT have epoch bits set
    uint8_t original[56];
    for (int i = 0; i < 56; ++i) {
        original[i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    }

    uint8_t epoch = 0xBB;

    uint64_t slot[8];
    pack_all_lanes(original, slot, epoch);

    // Lanes 0-6 should contain only payload bytes, no epoch contamination
    for (int lane = 0; lane < 7; ++lane) {
        uint8_t top_byte = static_cast<uint8_t>((slot[lane] >> 56) & 0xFF);
        size_t offset = static_cast<size_t>(lane) * 8 + 7;
        if (offset < 56) {
            EXPECT_EQ(top_byte, original[offset])
                << "Lane " << lane << " top byte should be payload, not epoch";
        }
    }

    // Lane 7 should have the epoch as full uint64_t
    EXPECT_EQ(get_epoch(slot[7]), epoch);
    EXPECT_EQ(slot[7], static_cast<uint64_t>(epoch));
}

// ============================================================
// Configuration tests
// ============================================================

TEST(ConfigTest, ValidPowerOf2) {
    // Various valid powers of 2
    size_t caps[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (size_t cap : caps) {
        EXPECT_NO_THROW(SPSCD2HQueue q(cap));
    }
}

TEST(ConfigTest, NotPowerOf2_Throws) {
    EXPECT_THROW(SPSCD2HQueue q(0), queue_error);
    EXPECT_THROW(SPSCD2HQueue q(3), queue_error);
    EXPECT_THROW(SPSCD2HQueue q(5), queue_error);
    EXPECT_THROW(SPSCD2HQueue q(100), queue_error);
    EXPECT_THROW(SPSCD2HQueue q(1023), queue_error);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
