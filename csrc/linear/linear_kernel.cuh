#pragma once

#include "linear.h"
#include "cutlass/gemm/device/gemm.h"

using TrainingForwardCTAShape = cutlass::gemm::GemmShape<128, 128, 8>;
using TrainingForwardWarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using TrainingForwardInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using BackwardCTAShape = cutlass::gemm::GemmShape<64, 64, 8>;
using BackwardWarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using BackwardInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using InferenceCTAShape = cutlass::gemm::GemmShape<8, 32, 32>;
// With K_tile=16 the FP32 SIMT transpose padding is 32/16=2.
// Warp<8,16,32> gives LaneM=2, LaneN=2, both dividing this padding.
// Warp<8,32,32> gives LaneN=4 and fails CUTLASS's padding assertion.
using InferenceWarpShape = cutlass::gemm::GemmShape<8, 16, 32>;
using InferenceInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

// This two-stage scalar SIMT path stores shared-memory fragments without a
// tile predicate. Each operand tile must distribute evenly over the CTA threads.
// CTA<8,32,8> / Warp<4,16,8> gives 128 threads for only 64 A elements,
// causing excess shared-memory stores to overlap valid A data.
static_assert(InferenceCTAShape::kM % InferenceWarpShape::kM == 0 &&
              InferenceCTAShape::kN % InferenceWarpShape::kN == 0 &&
              InferenceCTAShape::kK == InferenceWarpShape::kK,
              "Small-M SIMT tiles must divide evenly with no warp partition along K");
constexpr int kInferenceThreads = 32 *
    (InferenceCTAShape::kM / InferenceWarpShape::kM) *
    (InferenceCTAShape::kN / InferenceWarpShape::kN);
static_assert((InferenceCTAShape::kM * InferenceCTAShape::kK) % kInferenceThreads == 0 &&
              (InferenceCTAShape::kN * InferenceCTAShape::kK) % kInferenceThreads == 0,
              "Small-M SIMT operand tiles must contain a whole number of elements per CTA thread");

using OpClass = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm80;
using Operator = cutlass::arch::OpMultiplyAdd;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination< float, 1, float, float>;
using SwizzleOp = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr int kAlignmentA = 1;
constexpr int kAlignmentB = 1;
constexpr int kStages = 2;

using ForwardGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    TrainingForwardCTAShape,
    TrainingForwardWarpShape,
    TrainingForwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentA,
    kAlignmentB,
    false,
    Operator
    >;

using BackwardGemmX = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    BackwardCTAShape,
    BackwardWarpShape,
    BackwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentB,
    kAlignmentA,
    false,
    Operator
    >;

using BackwardGemmW = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    BackwardCTAShape,
    BackwardWarpShape,
    BackwardInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentB,
    kAlignmentA,
    false,
    Operator
    >;

using InferenceGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    OpClass,
    SmArch,
    InferenceCTAShape,
    InferenceWarpShape,
    InferenceInstructionShape,
    EpilogueOp,
    SwizzleOp,
    kStages,
    kAlignmentA,
    kAlignmentB,
    false,
    Operator
    >;
